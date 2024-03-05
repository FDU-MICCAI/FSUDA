import torch
import numpy as np
import torch.nn as nn



class GCloss(nn.Module):
    def __int__(self):
        super(GCloss, self).__init__()

    def forward(self, s_i, s_trans, t_trans):
        temp = 10
        # pos score
        # pos = torch.exp(torch.clip(torch.sum(s_i * s_trans, dim= -1), -80,80 )/ temp)
        pos = torch.exp(torch.sum(s_i * s_trans, dim=-1) / temp)
        # neg score
        # neg1 = torch.exp(torch.clip(torch.sum(s_i * s_j, dim= -1) ,-80, 80)/ temp)
        neg2 = torch.exp(torch.sum(s_i * t_trans, dim=-1) / temp)
        # neg2 = torch.exp(torch.clip(torch.sum(s_i * t_trans, dim= -1) ,-80, 80)/ temp)
        all = pos  + neg2

        # loss = - torch.log(torch.clip(pos / all, 1e-40).mean() - torch.log(torch.clip(1 - neg1 / all,1e-40)).mean() - torch.log(torch.clip(1 - neg2 / all, 1e-40))).mean()
        loss = - torch.log(torch.clip(pos / all, 1e-40)).mean() - torch.log(torch.clip(1 - neg2 / all, 1e-40)).mean()
        return loss


class LCloss(nn.Module):
    def __int__(self, temp):
        super(LCloss, self).__init__()
        self.temp = temp

    def forward(self, f_s, f_trans):
        temp = 10
        # q_s = f_s.reshape(f_s.shape[0],f_s.shape[1]*f_s.shape[2],f_s.shape[3])
        # q_trans = f_trans.reshape(f_trans.shape[0], f_trans.shape[1] * f_trans.shape[2], f_trans.shape[3])
        sim_matrix = torch.cosine_similarity(f_s.unsqueeze(2), f_trans.unsqueeze(1), dim=-1, eps=1e-08)
        sim_matrix_index = sim_matrix.max(dim=1)[1]

        pos_q_trans =  np.take_along_axis(f_trans,np.array(sim_matrix_index.cpu())[...,None],axis=1)
        pos = torch.multiply(f_s, pos_q_trans) / temp
        pos_exp = torch.exp(torch.sum(pos, dim=-1))
        all = torch.multiply(f_s.unsqueeze(2), f_trans.unsqueeze(1)) / temp
        all_exp = torch.sum(torch.exp(torch.sum(all, dim=-1)),dim=-1)
        loss = (torch.sum(- torch.log(torch.clip(torch.div(pos_exp, all_exp),1e-40)),dim=-1) / (f_s.shape[1])).mean()
        return loss

if __name__ == '__main__':
    s_i,s_j,s_trans,t_trans = torch.randn(4,128),torch.randn(4,128),torch.randn(4,128),torch.randn(4,128)
    f_s, f_trans = torch.randn(4,32,32,128), torch.randn(4,32,32,128)
    # GC = GCloss()
    # loss = GC(s_i,s_j,s_trans,t_trans)
    LC = LCloss()
    loss = LC(f_s, f_trans)




