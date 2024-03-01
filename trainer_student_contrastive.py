import time

import monai.metrics
import numpy as np
import random

import scipy.stats
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import copy
import log
import sys,os

from data.dataloader import datareader,PSUdatareader,SSLdatareader
from sklearn.metrics import jaccard_score
import logging
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils_t import *
from ssim import SSIM
from NSCT import NSCT_filter
# from skimage.exposure import match_histograms
from KD import  DistillKL
from contrastive_loss import GCloss, LCloss
def StyleRandomization(x, y, eps=1e-5):
    N, H, W = x.size()
    x = x.view(N, -1)
    mean_x = x.mean(-1, keepdim=True)
    var_x = x.var(-1, keepdim=True)

    # alpha = torch.rand(N, 1)
    alpha =0.1
    if x.is_cuda:
        alpha = alpha.cuda()

    y = y.view(N, -1)
    # idx_swap = torch.randperm(N)
    # y = y[idx_swap]

    # 1.use batch info
    mean_y = (y.mean(-1)).mean(-1)
    var_y = (y.var(-1)).mean(-1)

    # 2.single trgt sample info
    # mean_y = y.mean(-1, keepdim=True)
    # var_y = y.var(-1, keepdim=True)

    mean_fuse = alpha * mean_x + (1 - alpha) * mean_y
    var_fuse = alpha * var_x + (1 - alpha) * var_y

    x = (x - mean_x) / (var_x + eps).sqrt()
    x = x * (var_fuse + eps).sqrt() + mean_fuse
    x = x.view(N, H, W)

    return x, y.view(N, H, W)
def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# init
# ==================
def trainer_student_contrastive(args, model, snapshot_path):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('./checkpoints/' + args.exp_name):
        os.makedirs('./checkpoints/' + args.exp_name)
    if not os.path.exists('./checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('./checkpoints/' + args.exp_name + '/' + 'models')

    io = log.IOStream(args)
    # io.cprint(str(args))
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes

    # ==================
    # Read Data
    # ==================
    if args.train:

        src_trainset = datareader(args, args.dataroot, dataset=args.src_dataset + '_train', partition='train',domain='source')
        src_valset = datareader(args, args.dataroot, dataset=args.src_dataset + '_val', partition='val',domain='source')
        trgt_trainset = datareader(args, args.dataroot, dataset=args.trgt_dataset + '_train', partition='train',domain='target')
        trgt_valset = datareader(args, args.dataroot, dataset=args.trgt_dataset + '_val', partition='val',domain='target')
        src_train_loader = DataLoader(src_trainset, num_workers=0, batch_size=args.batch_size, shuffle=True,drop_last=True)
        src_val_loader = DataLoader(src_valset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
        trgt_train_loader = DataLoader(trgt_trainset, num_workers=0, batch_size=args.batch_size, shuffle=True,drop_last=True)
        trgt_val_loader = DataLoader(trgt_valset, num_workers=0, batch_size=args.batch_size, shuffle=True,drop_last=True)



    # ==================
    # Init Model
    # ==================
    writer = SummaryWriter(comment=args.exp_name)

    #Load multi-teacher pretrained models

    model_NSCT = model.to(device)
    model_hm = model.to(device)
    model = model.to(device)

    checkpoint = torch.load('/media/eric/DATA/Githubcode/medicalDA/FSUDA/checkpoints/_chaos_hm_mr_ct_b32_e50/models/model_best_trgt45.t7')
    model_NSCT.load_state_dict(checkpoint['model'])
    model_NSCT.eval()
    checkpoint = torch.load('/media/eric/DATA/Githubcode/medicalDA/FSUDA/checkpoints/_chaos_nsct03_mr_ct_b32_e50/models/model_best32.t7')
    model_hm.load_state_dict(checkpoint['model'])
    model_hm.eval()

    # Handle multi-gpu
    if (device.type == 'cuda') and len(args.gpus) > 1:
        model = nn.DataParallel(model, args.gpus)
    best_model = copy.deepcopy(model)

    # ==================
    # Optimizer
    # ==================

    opt_model = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" else optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.wd)
    t_max = args.epochs
    scheduler_model = CosineAnnealingLR(opt_model, T_max=t_max, eta_min=0.0)

    # ==================
    # Loss and Metrics
    # ==================

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    KL_loss = DistillKL(args.kd_T)
    GC_loss = GCloss()
    LC_loss = LCloss()

    if args.train == True:
        # =================
        # ------Train------
        # =================
        src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
        src_train_dice = src_train_dice_NSCT = src_train_dice_hm = trgt_train_dice = src_train_asd = trgt_train_asd = 0
        src_best_val_mIOU = trgt_best_val_mIOU = 0.0
        src_best_val_loss = trgt_best_val_loss = 90000000
        epoch =  0
        mean_img = torch.zeros(1, 1)
        for epoch in range(args.epochs):

            model.train()
            # init data structures for saving epoch stats
            src_seg_loss =  src_seg_loss_NSCT =  src_seg_loss_hm  = trgt_seg_loss = kl_loss = contrastive_loss = 0.0
            batch_idx = src_count = trgt_count = trgt_active_count = 0
            step = 0
            for data1, data2 in zip(src_train_loader, trgt_train_loader):
                step += 1
                opt_model.zero_grad()
                # opt_model_NSCT.zero_grad()
                # opt_model_hm.zero_grad()
                # opt_seg_NSCT.zero_grad()
                # opt_seg_hm.zero_grad()
                # opt_seg.zero_grad()

                ###source data###
                if data1 is not None:
                    src_in_trg, trg_in_src = [], []
                    src_data, src_labels, src_labels_orig = data1[0], data1[1].to(device), data1[2].permute(0, 3, 1,2).to(device)
                    trgt_data, trgt_labels, trgt_labels_orig = data2[0], data2[1].to(device), data2[2].permute(0, 3, 1,2).to(device)
                    # src_data, src_labels = src_data, src_labels
                    # trgt_data, trgt_labels, trgt_labels_orig = data2[0], data2[1].to(device), data2[2].permute(0, 3, 1,2).to(device)
                    batch_size = src_data.shape[0]
                    src_data = src_data.unsqueeze(1).to(device).type(torch.float32)
                    trgt_data = trgt_data.unsqueeze(1).to(device)


                    ###Fourier Style Augumentation
                    src2trgt, trgt2src = FDA_source_to_target_np(src_data,trgt_data)
                    src2trgt = torch.tensor(src2trgt).cuda().type(torch.float32)
                    trgt2src = torch.tensor(trgt2src).cuda().type(torch.float32)

                    ###student training###
                    outputs, encoded_features = model(src_data)

                    _, encoded_features1 = model(src2trgt)
                    _, encoded_features2 = model(trgt2src)
                    encoded_features_G = torch.mean(encoded_features,dim=1)
                    encoded_features1_G = torch.mean(encoded_features1,dim=1)
                    encoded_features2_G = torch.mean(encoded_features2, dim=1)
                    # encoded_features2 = encoded_features2.reshape(batch_size,16,16,768).permute(0,3,1,2)
                    with torch.no_grad():
                        outputs_NSCT,_ = model_NSCT(src_data)
                        outputs_hm,_ = model_hm(src_data)
                    loss_ce = ce_loss(outputs, src_labels[:].long())
                    loss_dice, classwise_dice = dice_loss(outputs, src_labels.long(), softmax=True, weight=args.diceweight)

                    ###Contrastive learning###
                    loss_gc = GC_loss(encoded_features_G,encoded_features1_G,encoded_features2_G)
                    loss_lc = LC_loss(encoded_features, encoded_features1)
                    loss_contrastive = 100 * loss_gc  + 0.2 * loss_lc

                    ###KL LOSS###
                    loss_div_list = [KL_loss(outputs, outputs_NSCT), KL_loss(outputs, outputs_hm)]
                    entropy_list = []
                    for logit_t in [outputs_NSCT,outputs_hm]:
                        softmax_logit_t = F.softmax(logit_t, dim=1)
                        entropy = -(softmax_logit_t) * (torch.log(softmax_logit_t))
                        entropy_list.append(entropy.sum(1))
                    weight = []
                    entropy_sum = torch.stack(entropy_list, dim=0).sum(dim=0).squeeze(dim=0)
                    for entropy in entropy_list:
                        weight.append(1.0 - (entropy / entropy_sum))
                    loss_div = torch.stack(loss_div_list)
                    weight = torch.stack(weight)
                    bsz = loss_div.shape[1]
                    loss_kl = (torch.mul(weight, loss_div).sum()) / (1.0 * bsz * 2)
                    # loss_kl = 0.2 * KL_loss(outputs, outputs_NSCT) + 0.8 * KL_loss(outputs, outputs_hm)

                    loss_seg = args.ceW * loss_ce + args.diceW * loss_dice + args.klW * loss_kl + args.ctW * loss_contrastive
                    loss_seg.backward()
                    opt_model.step()
                    src_seg_loss += loss_seg.item() * batch_size
                    kl_loss += loss_kl.item() * batch_size
                    contrastive_loss += loss_contrastive.item() * batch_size
                    src_train_dice += (1 - loss_dice.item()) * batch_size

                    src_count += batch_size
                    trgt_count += batch_size



                scheduler_model.step()

                src_seg_loss /= src_count
                src_train_dice /= src_count
                kl_loss /= src_count
                contrastive_loss /= src_count

            # ==================
            # Validation/test
            # ==================

            def validation(test_loader, model):

                # Run on cpu or gpu
                seg_loss = mIOU = accuracy = dice = asd = 0.0
                batch_idx = num_samples = 0
                back = AA = LAC = LVC = MYO = 0.0
                back_asd = AA_asd = LAC_asd = LVC_asd = MYO_asd = 0.0
                class_dice = []
                class_asd = []
                with torch.no_grad():
                    model.eval()

                    for i, data in enumerate(test_loader):
                        data, labels, labels_orig = data[0].to(device), data[1].to(device), data[2].to(device)
                        data = data.unsqueeze(1).type(torch.float32)
                        batch_size = data.shape[0]


                        outputs, _ = model(data)
                        loss_ce = ce_loss(outputs, labels[:].long())
                        loss_dice, classwise_dice = dice_loss(outputs, labels, softmax=True)
                        loss = 0.5 * loss_ce + 0.5 * loss_dice
                        seg_loss += loss.item() * batch_size
                        dice += (1 - loss_dice.item()) * batch_size
                        back += classwise_dice[0]
                        AA += classwise_dice[2]
                        LAC += classwise_dice[4]
                        LVC += classwise_dice[3]
                        MYO += classwise_dice[1]

                        # classwise_asd = ASD(outputs, labels_orig)
                        # back_asd += classwise_asd[0]
                        # AA_asd  += classwise_asd[2]
                        # LAC_asd += classwise_asd[4]
                        # LVC_asd += classwise_asd[3]
                        # MYO_asd += classwise_asd[1]

                        num_samples += batch_size
                        batch_idx += 1

                seg_loss /= num_samples
                dice /= num_samples
                back /= batch_idx
                AA /= batch_idx
                LAC /= batch_idx
                LVC /= batch_idx
                MYO /= batch_idx
                back_asd /= batch_idx
                AA_asd /= batch_idx
                LAC_asd /= batch_idx
                LVC_asd /= batch_idx
                MYO_asd /= batch_idx

                class_dice.append(back)
                class_dice.append(AA)
                class_dice.append(LAC)
                class_dice.append(LVC)
                class_dice.append(MYO)
                class_asd.append(back_asd)
                class_asd.append(AA_asd)
                class_asd.append(LAC_asd)
                class_asd.append(LVC_asd)
                class_asd.append(MYO_asd)

                return seg_loss, dice, class_dice, class_asd

            def test(test_loader, model):

                # Run on cpu or gpu
                seg_loss = mIOU = accuracy = dice = asd = 0.0
                batch_idx = num_samples = 0
                back = AA = LAC = LVC = MYO = 0.0
                back_asd = AA_asd = LAC_asd = LVC_asd = MYO_asd = 0.0
                class_dice = []
                class_asd = []
                with torch.no_grad():
                    model.eval()

                    for i, data in enumerate(test_loader):
                        data, labels, labels_orig, name = data[0].to(device), data[1].to(device), data[3].permute(0, 3,1,2).to(device), data[2]
                        data = data.unsqueeze(1)
                        batch_size = data.shape[0]

                        outputs,_ = model(data)

                        if labels_orig[:, 2, :, :].max() != 0:
                            print("")
                        pred = torch.softmax(outputs, dim=1)
                        index = pred.argmax(dim=1)
                        # map = one_hot_encoder(index)                loss_ce = ce_loss(outputs, labels[:].long())
                        loss_dice, classwise_dice = dice_loss(outputs, labels, softmax=True)
                        loss = args.ceW * loss_ce + args.diceW * loss_dice
                        seg_loss += loss.item() * batch_size
                        dice += (1 - loss_dice.item()) * batch_size
                        back += classwise_dice[0]
                        AA += classwise_dice[2]
                        LAC += classwise_dice[4]
                        LVC += classwise_dice[3]
                        MYO += classwise_dice[1]

                        classwise_asd = ASD(outputs, labels_orig)
                        back_asd += classwise_asd[0]
                        AA_asd += classwise_asd[2]
                        LAC_asd += classwise_asd[4]
                        LVC_asd += classwise_asd[3]
                        MYO_asd += classwise_asd[1]

                        num_samples += batch_size
                        batch_idx += 1

                seg_loss /= num_samples
                dice /= num_samples
                back /= batch_idx
                AA /= batch_idx
                LAC /= batch_idx
                LVC /= batch_idx
                MYO /= batch_idx
                back_asd /= batch_idx
                AA_asd /= batch_idx
                LAC_asd /= batch_idx
                LVC_asd /= batch_idx
                MYO_asd /= batch_idx

                class_dice.append(back)
                class_dice.append(AA)
                class_dice.append(LAC)
                class_dice.append(LVC)
                class_dice.append(MYO)
                class_asd.append(back_asd)
                class_asd.append(AA_asd)
                class_asd.append(LAC_asd)
                class_asd.append(LVC_asd)
                class_asd.append(MYO_asd)

                return seg_loss, dice, class_dice, class_asd

            # ===================
            # Validation
            # ===================

            src_val_loss, src_val_dice, src_val_classwise_dice, src_val_asd = validation(src_val_loader, model)
            trgt_val_loss, trgt_val_dice, trgt_val_classwise_dice, trgt_val_asd = validation(trgt_val_loader, model)


            # save model according to best source model (since we don't have target labels)
            if src_val_loss < src_best_val_loss:
                src_best_val_loss = src_val_loss
                # trgt_best_val_loss = trgt_val_loss
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)
            if trgt_val_loss < trgt_best_val_loss:
                trgt_best_val_loss = trgt_val_loss
                best_target_val_epoch = epoch
                best_target_model = copy.deepcopy(model)

            io.cprint(f"Epoch: {epoch}, "
                      f"Source train seg loss: {src_seg_loss:.5f}, "
                      f"Source train dice : {src_train_dice:.5f},"
                      f"Source kl loss : {kl_loss:.5f},"
                      f"Source ct loss : {contrastive_loss:.5f},"
                      # f"Source train asd : {src_train_asd:.5f}, \n"
                      # f"Target train seg loss: {trgt_seg_loss:.5f}, "
                      # f"Target train dice : {trgt_train_dice:.5f}, "
                      # f"Target train asd : {trgt_train_asd:.5f}, \n"
                      )
            io.cprint(f"Epoch: {epoch}, "
                      f"Source val seg loss: {src_val_loss:.5f}, "
                      f"Source val dice : {src_val_dice:.5f}, \n"
                      f"Source val classwise_dice(back,AA,LAC,LVC,MYO) : {src_val_classwise_dice[0]:.5f}, {src_val_classwise_dice[2]:.5f},{src_val_classwise_dice[4]:.5f},{src_val_classwise_dice[3]:.5f},{src_val_classwise_dice[1]:.5f}\n"
                      # f"Source val asd(back,AA,LAC,LVC,MYO) : {src_val_asd[0]:.5f}, {src_val_asd[2]:.5f},{src_val_asd[4]:.5f},{src_val_asd[3]:.5f},{src_val_asd[1]:.5f}"
                      # f"Source val hd(back,MYO,AA,LVA,LAC) : {src_val_hd[0]:.5f}, {src_val_hd[1]:.5f},{src_val_hd[2]:.5f},{src_val_hd[3]:.5f},{src_val_hd[4]:.5f}"
                      )
            io.cprint(f"Epoch: {epoch}, "
                      f"Target val seg loss: {trgt_val_loss:.5f}, "
                      f"Target val dice : {trgt_val_dice:.5f}, \n"
                      f"Target val classwise_dice(back,AA,LAC,LVC,MYO) : {trgt_val_classwise_dice[0]:.5f}, {trgt_val_classwise_dice[2]:.5f},{trgt_val_classwise_dice[4]:.5f},{trgt_val_classwise_dice[3]:.5f},{trgt_val_classwise_dice[1]:.5f} \n"
                      # f"Target val asd(back,AA,LAC,LVC,MYO) : {trgt_val_asd[0]:.5f}, {trgt_val_asd[2]:.5f},{trgt_val_asd[4]:.5f},{trgt_val_asd[3]:.5f},{trgt_val_asd[1]:.5f} "
                      # f"Target val asd(back,MYO,AA,LVA,LAC) : {trgt_val_hd[0]:.5f}, {trgt_val_hd[1]:.5f},{trgt_val_hd[2]:.5f},{trgt_val_hd[3]:.5f},{trgt_val_hd[4]:.5f}"
                      )


            writer.add_scalar('Source_train_seg/loss', src_seg_loss, epoch)
            writer.add_scalar('Source_train_seg/Dice', src_train_dice, epoch)
            writer.add_scalar('Source_train_seg/KL_loss', loss_kl, epoch)
            writer.add_scalar('Source_train_seg/CT_loss', loss_contrastive, epoch)
            writer.add_scalar('Target_train_seg/loss', trgt_seg_loss, epoch)
            writer.add_scalar('Source_val_seg/loss', src_val_loss, epoch)
            writer.add_scalar('Source_val_seg/Dice', src_val_dice, epoch)
            writer.add_scalar('Target_val_seg/loss', trgt_val_loss, epoch)
            writer.add_scalar('Target_val_seg/Dice', trgt_val_dice, epoch)

            # checkpoint = {
            #     "model": model.state_dict(),
            #     "seg": seg_head.state_dict(),
            #     'optimizer_model': opt_model.state_dict(),
            #     'optimizer_seg': opt_seg.state_dict(),
            #     "best_val_epoch": epoch,
            #     'lr_schedule_model': scheduler_model.state_dict(),
            #     'lr_schedule_seg': scheduler_seg.state_dict()
            # }
            # torch.save(checkpoint, './checkpoints/%s/models/model%s.t7' % (args.exp_name, str(epoch)))
        io.cprint("Best model was found at epoch %d\n"
                  "source val seg loss: %.4f"
                  "target val seg loss: %.4f\n"
                  % (best_val_epoch, src_best_val_loss, trgt_best_val_loss))
        checkpoint = {
            "model": best_model.state_dict(),
            'optimizer_model': opt_model.state_dict(),
            "best_val_epoch": epoch,
            'lr_schedule_model': scheduler_model.state_dict()
        }
        torch.save(checkpoint, './checkpoints/%s/models/model_best%s.t7' % (args.exp_name, str(best_val_epoch)))
        checkpoint = {
            "model": best_target_model.state_dict(),
            'optimizer_model': opt_model.state_dict(),
            "best_target_val_epoch": epoch,
            'lr_schedule_model': scheduler_model.state_dict()
        }

    # ===================
    # May be useful
    # ===================
    # if args.rotate == True:
    #     B, R, C, H, W = src_rotate_data.shape
    #     src_rotate_data = src_rotate_data.view(B * R, C, H, W)
    #     x, features = model(src_rotate_data)
    #     cls_pred = cls_head(x)
    #     loss_cls_src = ce_loss(cls_pred, src_rotate_label.view(-1).long())
    #     loss_cls_src.backward()
    #     opt_model.step()
    #     opt_rotate.step()
    #     # print('loss_cls_src:',loss_cls_src.item())
    #     src_cls_loss += loss_cls_src.item() * batch_size
    #
    #     B, R, C, H, W = trgt_rotate_data.shape
    #     trgt_rotate_data = trgt_rotate_data.view(B * R, C, H, W)
    #     x, features = model(trgt_rotate_data)
    #     cls_pred = cls_head(x)
    #     loss_cls_trgt = ce_loss(cls_pred, trgt_rotate_label.view(-1).long())
    #     loss_cls_trgt.backward()
    #     opt_model.step()
    #     opt_rotate.step()
    #     # print('loss_cls_trgt:',loss_cls_trgt.item())
    #     trgt_cls_loss += loss_cls_trgt.item() * batch_size


    ####huitu
    # import matplotlib
    # colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', '#ff9393']
    # c = matplotlib.colors.ListedColormap(colors)
    # for i in range(data.shape[0]):
    #     plt.imshow(data[i, 0, :, :].detach().cpu(), cmap='gray')
    #     plt.show()
    #     plt.imshow(labels[i, :, :].detach().cpu(), cmap=c)
    #     plt.show()
    #     plt.imshow(index[i, :, :].detach().cpu(), cmap=c)
    #     plt.show()
    ####save
    # i = 4
    # m = 0
    # plt.imshow(data[m, 0, :, :].detach().cpu(), cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_image%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(labels[m, :, :].detach().cpu(), cmap=c)
    # plt.axis('off')
    # plt.savefig('ct_label%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(index[m, :, :].detach().cpu(), cmap=c)
    # plt.axis('off')
    # plt.savefig('ct_pred%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # print(name[m])

    # i = 4
    # m = 7
    # plt.imshow(data[m, 0, :, :].detach().cpu(), cmap='gray')
    # plt.axis('off')
    # plt.savefig('mr_image%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(labels[m, :, :].detach().cpu(), cmap=c)
    # plt.axis('off')
    # plt.savefig('mr_label%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(index[m, :, :].detach().cpu(), cmap=c)
    # plt.axis('off')
    # plt.savefig('mr_pred%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # print(name[m])
#purple
    # import matplotlib
    #
    # colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', ]
    # c = matplotlib.colors.ListedColormap(colors)
    # plt.imshow(labels[0, :, :].detach().cpu(), cmap=c)
    # plt.show()
    # plt.imshow(index[0, :, :].detach().cpu(), cmap=c)
    # plt.show()
####save spectrum
    # plt.imshow(A2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_A%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(H2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_H%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(V2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_V%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(D2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_D%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)

###mri
# import matplotlib
# colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', '#ff9393']
# c = matplotlib.colors.ListedColormap(colors)
# for i in range(6):
#     plt.imshow(index[i, :, :].detach().cpu(), cmap=c)
#     plt.axis('off')
#     plt.savefig('mr1_%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)