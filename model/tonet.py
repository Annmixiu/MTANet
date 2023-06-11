"""
Add comments and rewrite on the basis of Ke Chen knutchen@ucsd.edu

This file contains:
A. the train/valid and test
B. tonet-decoder

"""

import os
import numpy as np
import torch
from torch import nn
from numpy import *
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from util import melody_eval, freq2octave, freq2tone, tofreq
from .attention_layer import CombineLayer, PositionalEncoding
from feature_extraction import get_CenFreq


class TONet(pl.LightningModule):
    """
    Args:
        mode: ["disable", "enable"]
    Annotation(mode):
    single:

    """

    def __init__(self, l_model, r_model, config, loss_func, mode="single"):
        super().__init__()
        self.config = config
        # l_model for original-CFP
        self.l_model = l_model
        # r_model for Tone-CFP
        self.r_model = r_model
        self.mode = mode
        self.centf = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
        self.centf[0] = 0
        self.loss_func = loss_func
        self.max_metric = np.zeros((3, 6))#three datasets_six metric
        if self.mode == "all" or self.mode == "tcfp":
            assert r_model is not None, "Enabling TONet needs two-branch models!"

        self.gru_dim = 512
        self.attn_dim = 2048#原先是2048
        # define hyperparameter
        if self.mode == "tcfp":
            self.sp_dim = self.config.freq_bin * 2
            self.linear_dim = self.config.freq_bin * 2
        elif self.mode == "spl":
            self.sp_dim = self.config.freq_bin
            self.linear_dim = self.gru_dim * 2
        elif self.mode == "spat":
            self.sp_dim = self.config.freq_bin
            self.linear_dim = self.attn_dim
        elif self.mode == "all":
            self.sp_dim = self.config.freq_bin * 2
            self.linear_dim = self.attn_dim

        # Network Architecture
        if self.mode == "spl":
            self.tone_gru = nn.Linear(self.sp_dim, self.linear_dim)
            # nn.GRU(
            # self.sp_dim, self.gru_dim, 1,
            # batch_first=True, bidirectional=True
            # )
            self.octave_gru = nn.Linear(self.sp_dim, self.linear_dim)
            # nn.GRU(
            #     self.sp_dim, self.gru_dim, 1,
            #     batch_first=True, bidirectional=True
            # )
        elif self.mode == "spat" or self.mode == "all":
            self.tone_in = nn.Linear(self.sp_dim, self.attn_dim)
            self.tone_posenc = PositionalEncoding(self.attn_dim, n_position=self.config.seg_frame)
            self.tone_dropout = nn.Dropout(p=0.2)
            self.tone_norm = nn.LayerNorm(self.attn_dim, eps=1e-6)
            self.tone_attn = nn.ModuleList([
                CombineLayer(self.attn_dim, 2.66, 8,
                             self.attn_dim // 8, self.attn_dim // 8, dropout=0.2)
                for _ in range(2)]
            )
            self.octave_in = nn.Linear(self.sp_dim, self.attn_dim)
            self.octave_posenc = PositionalEncoding(self.attn_dim, n_position=self.config.seg_frame)
            self.octave_dropout = nn.Dropout(p=0.2)
            self.octave_norm = nn.LayerNorm(self.attn_dim, eps=1e-6)
            self.octave_attn = nn.ModuleList([
                CombineLayer(self.attn_dim, 2.66, 8,
                             self.attn_dim // 8, self.attn_dim // 8, dropout=0.2)
                for _ in range(2)]
            )
        if self.mode != "single" and self.mode != "tcfp":
            # Guess fully-connected layers
            self.tone_linear = nn.Sequential(
                nn.Linear(self.linear_dim, 512),
                nn.Dropout(p=0.2),
                nn.SELU(),
                nn.Linear(512, 128),
                nn.Dropout(p=0.2),
                nn.SELU(),
                nn.Linear(128, self.config.tone_class),
                nn.Dropout(p=0.2),
                nn.SELU()
            )
            self.octave_linear = nn.Sequential(
                nn.Linear(self.linear_dim, 256),
                nn.Dropout(p=0.2),
                nn.SELU(),
                nn.Linear(256, 64),
                nn.Dropout(p=0.2),
                nn.SELU(),
                nn.Linear(64, self.config.octave_class),
                nn.Dropout(p=0.2),
                nn.SELU()
            )
            self.tone_bm = nn.Sequential(
                nn.Linear(2, 1),
                nn.SELU()
            )
            self.octave_bm = nn.Sequential(
                nn.Linear(2, 1),
                nn.SELU()
            )
            # [bs, 361 + 13 + 9, 128]
            self.tcfp_linear = nn.Sequential(
                nn.Conv1d(self.config.freq_bin * 2, self.config.freq_bin,#将频率维度看做C，沿时间维度作conv1D后叠加
                          5, padding=2),
                nn.SELU()
            )
            self.tcfp_bm = nn.Sequential(
                nn.Conv1d(2, 1, 5, padding=2),
                nn.SELU()
            )
            self.final_linear = nn.Sequential(
                nn.Conv1d(
                    self.config.tone_class + self.config.octave_class + self.config.freq_bin + 3,
                    self.config.freq_bin, 5, padding=2),
                nn.SELU()
            )
        elif self.mode == "tcfp":
            self.final_linear = nn.Sequential(
                nn.Linear(self.linear_dim, self.config.freq_bin),# dimension reduction
                nn.SELU()
            )
            self.final_bm = nn.Sequential(
                nn.Linear(2, 1),
                nn.SELU()
            )

    """
    Args:
        x: [bs, 3, freuqncy_bin, time_frame]
    """

    def tone_decoder(self, tone_feature):
        if self.mode == "all" or self.mode == "spat":
            tone_h = self.tone_dropout(self.tone_posenc(self.tone_in(tone_feature)))
            tone_h = self.tone_norm(tone_h)
            for tone_layer in self.tone_attn:
                tone_h, tone_weight = tone_layer(tone_h, slf_attn_mask=None)
                #tone_h = feed forward_output=enc_output=q=[batch,q_len,d_model]=[b,128,2048]
                #tone_weight = MultiHeadAttention_feature=each_head_attention(Q,K,V)=[batch,n_head,len_q,len_k]
            tone_prob = self.tone_linear(tone_h)#fully-connected layers[b,128,12]
            tone_prob = tone_prob.permute(0, 2, 1).contiguous()#dimension_change and tensor_check[b,12,128]
        elif self.mode == "spl":#decode the tone/octave into presence probability maps
            tone_h = self.tone_gru(tone_feature)#[bs,128,1024]
            tone_prob = self.tone_linear(tone_h)#[bs,128,12]
            tone_prob = tone_prob.permute(0, 2, 1).contiguous()#[bs,12,128]
        return tone_prob

    def octave_decoder(self, octave_feature):
        if self.mode == "all" or self.mode == "spat":
            octave_h = self.octave_dropout(self.octave_posenc(self.octave_in(octave_feature)))
            octave_h = self.octave_norm(octave_h)
            for octave_layer in self.octave_attn:
                octave_h, octave_weight = octave_layer(octave_h, slf_attn_mask=None)
            octave_prob = self.octave_linear(octave_h)
            octave_prob = octave_prob.permute(0, 2, 1).contiguous()
        elif self.mode == "spl":
            octave_h = self.octave_gru(octave_feature)
            octave_prob = self.octave_linear(octave_h)
            octave_prob = octave_prob.permute(0, 2, 1).contiguous()
        return octave_prob

    def forward(self, x, tx=None):
        if self.mode == "single":
            output, _ = self.l_model(x)
            return output
        elif self.mode == "all":
            _, output_l = self.l_model(x)#[bs,1,361,128]
            _, output_r = self.r_model(tx)#[bs,1,361,128]
            bm_l = output_l[:, :, 0, :].unsqueeze(dim=2)#voice bottom,[bs,1,1,128]
            output_l = output_l[:, :, 1:, :]#FeatureMap,[bs,1,360,128]
            bm_r = output_r[:, :, 0, :].unsqueeze(dim=2)#[bs,1,1,128](b,c,f,t)
            output_r = output_r[:, :, 1:, :]#[bs,1,360,128]
            feature_agg = torch.cat((output_l, output_r), dim=2)#[bs,1,720,128]
            feature_agg = feature_agg.squeeze(dim=1)#[bs,720,128]
            # print("this is the feature",feature_agg.shape)
            feature_agg_mi = self.tcfp_linear(feature_agg)  # [bs, 360, 128]
            # 与文章中不同的一点，这里2F又做了线性变换，所以后面通过线性层后才传入最后特征融合的模块
            # 特征维度=[bs, 360, 128],但传入decoder的还是[bs,720,128]
            bm_agg = torch.cat((bm_l, bm_r), dim=2)#[bs,1,2,128]
            bm_agg = bm_agg.squeeze(dim=1)#[bs,2,128]
            bm_agg_mi = self.tcfp_bm(bm_agg)#[bs,1,128]
            bm_agg = bm_agg.permute(0, 2, 1)#[bs,128,2](b,t,f)
            tone_feature = feature_agg.permute(0, 2, 1).contiguous()#[bs,128,720]
            octave_feature = feature_agg.permute(0, 2, 1).contiguous()#[bs,128,720]
            tone_prob = self.tone_decoder(tone_feature)#[bs,12,128]
            octave_prob = self.octave_decoder(octave_feature)#[bs,8,128]

            tone_bm = self.tone_bm(bm_agg)#[bs,128,1]
            octave_bm = self.octave_bm(bm_agg)#[bs,128,1]
            tone_bm = tone_bm.permute(0, 2, 1)#[bs,1,128]，guess nonmelodic component
            octave_bm = octave_bm.permute(0, 2, 1)#[bs,1,128]，guess nonmelodic component

            tone_prob = torch.cat((tone_prob, tone_bm), dim=1)#[bs,13,128]
            octave_prob = torch.cat((octave_prob, octave_bm), dim=1)#[bs,9,128]

            final_feature = torch.cat((tone_prob, octave_prob, feature_agg_mi, bm_agg_mi), dim=1)#[bs,13+9+360+1,128]
            final_feature = self.final_linear(final_feature)#[bs,360,128],Time-Axis 1D-CNN
            final_feature = torch.cat((bm_agg_mi, final_feature), dim=1)#[bs,361,128]
            final_feature = nn.Softmax(dim=1)(final_feature)#[bs,361,128]
            tone_prob = nn.Softmax(dim=1)(tone_prob)#[bs,13,128]
            octave_prob = nn.Softmax(dim=1)(octave_prob)#[bs,9,128]
            return tone_prob, octave_prob, final_feature
        elif self.mode == "tcfp":
            _, output_l = self.l_model(x)#[bs,1,361,128]
            _, output_r = self.r_model(tx)#[bs,1,361,128]
            bm_l = output_l[:, :, 0, :].unsqueeze(dim=2)#[bs,1,1,128]
            output_l = output_l[:, :, 1:, :]#[bs,1,360,128]
            bm_r = output_r[:, :, 0, :].unsqueeze(dim=2)#[bs,1,1,128]
            output_r = output_r[:, :, 1:, :]#[bs,1,360,128]
            feature_agg = torch.cat((output_l, output_r), dim=2)#[bs,1,720,128]
            feature_agg = feature_agg.permute(0, 1, 3, 2)#[bs,1,128,720]
            bm_agg = torch.cat((bm_l, bm_r), dim=2)#[bs,1,2,128]
            bm_agg = bm_agg.permute(0, 1, 3, 2)#[bs,1,128,2]
            final_x = self.final_linear(feature_agg)#[bs,1,128,360]
            final_bm = self.final_bm(bm_agg)#[bs,1,128,1]
            final_x = final_x.permute(0, 1, 3, 2)#[bs,1,360,128]
            final_bm = final_bm.permute(0, 1, 3, 2)#[bs,1,1,128]
            final_output = nn.Softmax(dim=2)(torch.cat((final_bm, final_x), dim=2))#[bs,1,361,128]
            return final_output
        elif self.mode == "spl" or self.mode == "spat":
            _, output_l = self.l_model(x)#[bs,1,361,128]
            bm_l = output_l[:, :, 0, :].unsqueeze(dim=2)#[bs,1,1,128]
            output_l = output_l[:, :, 1:, :]#[bs,1,360,128]
            feature_agg = output_l#[bs,1,360,128]
            feature_agg = feature_agg.squeeze(dim=1)#[bs,360,128]
            bm_agg = bm_l#[bs,1,1,128]
            bm_agg = bm_agg.squeeze(dim=1)#[bs,1,128]
            tone_feature = feature_agg.permute(0, 2, 1).contiguous()#[bs,128,360]
            octave_feature = feature_agg.permute(0, 2, 1).contiguous()#[bs,128,360]
            tone_prob = self.tone_decoder(tone_feature)#[bs,12,128],spl/spat_output_dim is same
            octave_prob = self.octave_decoder(octave_feature)#[bs,8,128]
            tone_bm = bm_agg#[bs,1,128]
            octave_bm = bm_agg#[bs,1,128]

            tone_prob = torch.cat((tone_prob, tone_bm), dim=1)#[bs,12+1,128]=[bs,13,128]
            octave_prob = torch.cat((octave_prob, octave_bm), dim=1)#[bs,8+1,128]=[bs,9,128]

            final_feature = torch.cat((tone_prob, octave_prob, feature_agg, bm_agg), dim=1)#[bs,13+9+360+1,128]
            final_feature = self.final_linear(final_feature)#[bs,360,128]
            final_feature = torch.cat((bm_agg, final_feature), dim=1)#[bs,1+360,128]
            final_feature = nn.Softmax(dim=1)(final_feature)#[bs,361,128]
            tone_prob = nn.Softmax(dim=1)(tone_prob)#[bs,13,128]
            octave_prob = nn.Softmax(dim=1)(octave_prob)#[bs,9,128]
            return tone_prob, octave_prob, final_feature

    """
    Args:
        batch: {
            "cfp": [bs, 3, frequency_bin, time_frame],
            "gd": [bs, time_frame]
        }
    """

    def training_step(self, batch, batch_idx):
        device_type = next(self.parameters()).device
        # Specify the device that uses the same parameters as the data fed into the training model
        cfps = batch["cfp"]
        tcfps = batch["tcfp"]
        gds = batch["gd"]  # bs of (1*128)matrix，num_dim=T=128,content=fre_per_frame
        if self.mode == "single":
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to(device_type)
            # gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2], cfps.shape[-1])).to(device_type)
            for i in range(len(gds)):
                # len[bs, time_frame], returns the outermost dimension (b) size,i can iterate over batch_size
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            output = self(cfps)
            output = torch.squeeze(output, dim=1)
            loss = self.loss_func(output, gd_maps)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.print("trainable params:", param_list)
        elif self.mode == "all":
            # from pure pitch estimation,gds=bs of (1*128) matrix，num_dim=T=128,content=fre_per_frame
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to(device_type)#(bs,f+1，128)
            tone_maps = torch.zeros((cfps.shape[0], self.config.tone_class + 1, cfps.shape[-1])).to(device_type)#(bs,13,128)
            octave_maps = torch.zeros((cfps.shape[0], self.config.octave_class + 1, cfps.shape[-1])).to(device_type)#(bs,9,128)
            tone_index = ((gds % 60) * self.config.tone_class / 60).long()
            octave_index = (gds // 60 + 2).long()
            tone_index[gds < 1.0] = self.config.tone_class  # All f=0 frames are assigned the value 12
            octave_index[gds < 1.0] = self.config.octave_class  # All f=0 frames are assigned an 8
            for i in range(len(tone_maps)):
                # i traverses bs(0,1),len(tensor)= the most outer dim
                # binaryzation
                tone_maps[i, tone_index[i], torch.arange(gds.shape[-1])] = 1.0
                octave_maps[i, octave_index[i], torch.arange(gds.shape[-1])] = 1.0
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            tone_prob, octave_prob, final_prob = self(cfps, tcfps)
            pred_map = torch.cat((tone_prob, octave_prob, final_prob), dim=1)
            gd_map = torch.cat([tone_maps, octave_maps, gd_maps], dim=1)
            loss = self.loss_func(pred_map, gd_map)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.mode == "tcfp":
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to(device_type)
            for i in range(len(gds)):
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            output = self(cfps, tcfps)
            output = torch.squeeze(output, dim=1)
            loss = self.loss_func(output, gd_maps)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.mode == "spl" or self.mode == "spat":
            # from pure pitch estimation
            gd_maps = torch.zeros((cfps.shape[0], cfps.shape[-2] + 1, cfps.shape[-1])).to(device_type)
            tone_maps = torch.zeros((cfps.shape[0], self.config.tone_class + 1, cfps.shape[-1])).to(device_type)
            octave_maps = torch.zeros((cfps.shape[0], self.config.octave_class + 1, cfps.shape[-1])).to(device_type)
            tone_index = ((gds % 60) * self.config.tone_class / 60).long()
            octave_index = (gds // 60 + 2).long()
            tone_index[gds < 1.0] = self.config.tone_class
            octave_index[gds < 1.0] = self.config.octave_class
            for i in range(len(tone_maps)):
                tone_maps[i, tone_index[i], torch.arange(gds.shape[-1])] = 1.0
                octave_maps[i, octave_index[i], torch.arange(gds.shape[-1])] = 1.0
                gd_maps[i, gds[i].long(), torch.arange(gds.shape[-1])] = 1.0
            tone_prob, octave_prob, final_prob = self(cfps)
            pred_map = torch.cat((tone_prob, octave_prob, final_prob), dim=1)
            gd_map = torch.cat([tone_maps, octave_maps, gd_maps], dim=1)
            loss = self.loss_func(pred_map, gd_map)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def write_prediction(self, pred, filename):  # write_prediction(time-frequence file)
        time_frame = np.arange(len(pred)) * 0.01
        with open(filename, "w") as f:
            for i in range(len(time_frame)):
                # i=0,...,len(pred)-1
                f.write(str(np.round(time_frame[i], 4)) + "\t" + str(pred[i]) + "\n")

    def validation_step(self, batch, batch_idx, dataset_idx):
        # one interation
        # validate performance by processing data cross mode in the way of  bs=2
        device_type = next(self.parameters()).device
        mini_batch = self.config.batch_size
        cfps = batch["cfp"][0]  # (17,3,360,128),(25,3,360,128),(14,3,360,128)
        tcfps = batch["tcfp"][0]  # (17,3,360,128),(25,3,360,128),(14,3,360,128)
        gds = batch["gd"][0]  # (17,128),(25,128),(14,128)
        lens = batch["length"][0]  # batch["length"]=array([2109/3197/1748]),lens=array(2109/3197/1748)
        if self.mode == "single":
            output = []
            for i in range(0, len(cfps), mini_batch):
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                temp_output = self(temp_cfp)
                temp_output = torch.squeeze(temp_output, dim=1)
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(np.array(output), axis=0)
            return [
                output,
                gds,
                lens
            ]
        elif self.mode == "all":
            # output_tone = []
            # output_octave = []
            output = []
            for i in range(0, len(cfps), mini_batch):  # mini_batch=bs
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                # cfp:(17/25/14,3,360,128)推测首维是num_batch per_epoch
                # 首维以2步幅进行tensor切片,对于i=0，temp_cfp=(2,3,360,128)
                temp_tcfp = torch.from_numpy(tcfps[i:i + mini_batch]).to(device_type)  # same as above
                _, _, temp_output = self(temp_cfp, temp_tcfp)  # self amount to mode_l/r + mode
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(output, axis=0)  # all batch output(17,3,360,128)
            return [
                output,  # out_feature map
                gds,  # groundtruth
                lens  # gross time_frame length
            ]
        elif self.mode == "tcfp":
            output = []
            for i in range(0, len(cfps), mini_batch):
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                temp_tcfp = torch.from_numpy(tcfps[i:i + mini_batch]).to(device_type)
                temp_output = self(temp_cfp, temp_tcfp)
                temp_output = torch.squeeze(temp_output, dim=1)
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(np.array(output), axis=0)
            return [
                output,
                gds,
                lens
            ]
        elif self.mode == "spl" or self.mode == "spat":
            # output_tone = []
            # output_octave = []
            output = []
            for i in range(0, len(cfps), mini_batch):
                temp_cfp = torch.from_numpy(cfps[i:i + mini_batch]).to(device_type)
                _, _, temp_output = self(temp_cfp)
                temp_output = temp_output.detach().cpu().numpy()
                output.append(temp_output)
            output = np.concatenate(output, axis=0)
            return [
                output,
                gds,
                lens
            ]

    def validation_epoch_end(self, validation_step_outputs): # one epoch output in the form of list includes per_inter return
        if self.mode == "single" or self.mode == "tcfp":
            for i, dataset_d in enumerate(validation_step_outputs):
                metric = np.array([0., 0., 0., 0., 0., 0.])
                preds = []
                gds = []
                for d in dataset_d:
                    pred, gd, rl = d
                    pred = np.argmax(pred, axis=1)
                    pred = np.concatenate(pred, axis=0)
                    pred = self.centf[pred]
                    gd = np.concatenate(gd, axis=0)
                    preds.append(pred)
                    gds.append(gd)
                preds = np.concatenate(preds, axis=0)
                gds = np.concatenate(gds, axis=0)
                metric = melody_eval(preds, gds)
                self.print("\n")
                self.print("Dataset ", i, " OA:", metric[-1])
                if metric[-1] > self.max_metric[i, -1]:
                    for j in range(len(self.max_metric[i])):
                        self.max_metric[i, j] = metric[j]
                        self.max_metric[i, j] = metric[j]
                    torch.save(self.state_dict(), "model_backup/bestk_" + str(i) + ".ckpt")
                self.print("Best ", i, ":", self.max_metric[i])
        elif self.mode == "all" or self.mode == "spl" or self.mode == "spat":
            for i, dataset_d in enumerate(validation_step_outputs): # i is key(which datasets),dataset_d is value(pred,gd,len)
                metric = np.array([0., 0., 0., 0., 0., 0.])
                preds = []
                gds = []
                for d in dataset_d:
                    pred, gd, rl = d
                    # pred:(17/25/14,361,128)
                    # gd:(17/25/14,128)
                    # rl is 2109/3197/1748
                    pred = np.argmax(pred, axis=1)  # get f_dim max_index(which frequency bin per_frame is activated)
                    # along fre_dim(row),which columns are activated
                    # pred:(17/25/14,128)
                    pred = np.concatenate(pred, axis=0)  # if one para,joint along axis
                    # axis=0,,joint along row,pred:(17/25/14*128)=(2176/3200/1792)
                    pred = self.centf[pred]  # return list[32HZ,...,2048HZ],per_frame central_freq
                    gd = np.concatenate(gd, axis=0)  # (17/25/14,128)→(2176/3200/1792)
                    preds.append(pred)  # convert to list
                    gds.append(gd)  # convert to list
                preds = np.concatenate(preds, axis=0)  # result per_epoch
                gds = np.concatenate(gds, axis=0)  # result per_epoch
                metric = melody_eval(preds, gds)
                self.print("\n")
                self.print("Dataset ", i, " OA:", metric[-1])
                if metric[-1] > self.max_metric[i, -1]:
                    for j in range(len(self.max_metric[i])):  # j=0,1,2,3,4,5
                        self.max_metric[i, j] = metric[j]
                        self.max_metric[i, j] = metric[j]  # why repeati it?
                    torch.save(self.state_dict(), "model_backup/bestk_" + str(i) + ".ckpt")  # save mode state
                self.print("Best ", i, ":", self.max_metric[i])


    def test_step(self, batch, batch_idx, dataset_idx):
        # return is necessary
        return self.validation_step(batch, batch_idx, dataset_idx)

    def test_epoch_end(self, test_step_outputs):  # no return is required
        self.validation_epoch_end(test_step_outputs)
        for i, dataset_d in enumerate(test_step_outputs):
            for j, d in enumerate(dataset_d):
                pred, _, rl = d
                pred = np.argmax(pred, axis = 1)
                pred = np.concatenate(pred, axis = 0)[:rl]
                pred = self.centf[pred]
                self.write_prediction(pred, "prediction/" + str(i) + "_" + str(j) + ".txt")

    def configure_optimizers(self):
        # setting optimizer and parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)  # lr=initial lr

        # def lr_foo(epoch):
        #     if epoch < 10:
        #         # warm up lr
        #         lr_scale = 0.5
        #     elif epoch >= 10 and epoch < 30:
        #         lr_scale = 0.5 * 0.9 ** (epoch - 5)
        #     else:
        #         lr_scale = 0.04 * 0.92 ** (epoch - 30)
        #     return lr_scale

        def lr_foo(epoch):
            if epoch < 5:
                # warm up lr
                lr_scale = 0.5
            else:
                # lr_scale = 0.5 * 0.95 ** (epoch - 5)
                lr_scale = 0.5 * 0.98 ** (epoch - 5)

            return lr_scale

        if self.mode == "single" or self.mode == "tcfp":
            return optimizer

        elif self.mode == "all" or self.mode == "spl" or self.mode == "spat":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_foo  # attenuation strategy
            )
            return [optimizer], [scheduler]





