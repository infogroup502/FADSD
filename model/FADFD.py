import torch
import torch.nn as nn
import time



# class Encoder(nn.Module):
#     def __init__(self, attn_layers, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.norm = norm_layer
#
#     def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
#         series_list = []
#         prior_list = []
#         for attn_layer in self.attn_layers:
#             series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
#             series_list.append(series)
#             prior_list.append(prior)
#         return series_list, prior_list

time_start = time.time()

class FADFD(nn.Module):
    def __init__(self, p,select):
        super(FADFD, self).__init__()
        self.p =p
        self.select = select


    def forward(self, x, data_global):
        ori_x_1 = x.clone()
        ori_mean_1 = x.clone()

        ori_x_2 = data_global.clone()
        ori_mean_2 = data_global.clone()

        mean_val = torch.mean(x, dim=1)
        middle_index = x.size(1) // 2
        ori_mean_1[:, middle_index,  :] = mean_val

        mean_val = torch.mean(data_global, dim=1)
        middle_index = data_global.size(1) // 2
        ori_mean_2[:, middle_index,:,:] = mean_val

        ori_x_1 = ori_x_1[:, 0:-1, :]
        ori_x_2 = ori_x_2[:, :, 0:-1, :]
        ori_mean_1 = ori_mean_1[:, 0:-1, :]
        ori_mean_2 = ori_mean_2[:,:, 0:-1, :]

        in_x_1 = torch.fft.rfft(ori_x_1, dim=1, norm='ortho')
        in_mean_1_pre = torch.fft.rfft(ori_mean_1, dim=1, norm='ortho')
        in_x_2 = torch.fft.rfft(ori_x_2, dim=2, norm='ortho')
        in_mean_2_pre = torch.fft.rfft(ori_mean_2, dim=2, norm='ortho')



        if(self.select==0):
            score_1 = self.mse_1(torch.abs(in_mean_1_pre), torch.abs(in_x_1))
            score_2 = self.mse_2(torch.abs(in_mean_2_pre), torch.abs(in_x_2))
        else:
            score_1 = self.mse_1(torch.angle(in_mean_1_pre), torch.angle(in_x_1))
            score_2 = self.mse_2(torch.angle(in_mean_2_pre), torch.angle(in_x_2))

        score_1 = (score_1 - score_1.min()) / (score_1.max() - score_1.min())
        score_2 = (score_2 - score_2.min()) / (score_2.max() - score_2.min())

        score, _ = torch.max(torch.stack((score_1, score_2), dim=0), dim=0)

        return self.p*score_1+(1-self.p)*score_2

    def mse_1(self, x, y):
        return torch.mean(torch.mean((x-y)**2,dim=-1),dim=-1)

    def mse_2(self, x, y):
        return torch.mean(torch.mean(torch.mean((x-y)**2,dim=-1),dim=-1),dim=-1)


