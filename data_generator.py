"""
Rewriting on the basis of Ke Chen knutchen@ucsd.edu

This file contains:
A. the dataset and data generator classes
B. comments

"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from util import index2centf
from feature_extraction import get_CenFreq


# The constructs of TCFP are retained but not used
def reorganize(x, octave_res):
    # x:(3,360,128),octave_res:60
    n_order = []
    max_bin = x.shape[1]  # 360
    for i in range(octave_res):
        # i=0,1,...,59
        n_order += [j for j in range(i, max_bin, octave_res)]  # n_order is index of harmonics
        # n_order:(6*60)=[0,60,120,180,240,300,1,61,121,181,241,301,...,59,119,179,239,299,359]
    nx = [x[:, n_order[i], :] for i in range(x.shape[1])]  # i=0,1,...,359
    # nx=list[tensor0(3,128),tensor1(3,128),...,tenser359(3,128)]
    nx = np.array(nx)  # list to array(360,3,128)
    nx = nx.transpose((1, 0, 2))  # nx:(3,360,128)
    return nx


class TrainDataset(Dataset):
    def __init__(self, data_list, config):
        self.config = config
        self.cfp_dir = os.path.join(config.data_path, config.cfp_dir)
        self.f0_dir = os.path.join(config.data_path, "f0ref")
        self.data_list = data_list
        self.cent_f = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
        # init data array
        self.data_cfp = []
        self.data_gd = []
        self.data_tcfp = []
        self.data_len = []
        seg_frame = config.seg_frame
        shift_frame = config.shift_frame
        print("Data List:", data_list)
        with open(data_list, "r") as f:
            data_txt = f.readlines()
            data_txt = [d.split(".")[0] for d in data_txt]  # data_txt=[name1,name2,,...,last_name]
        # data_txt = data_txt[:100]
        print("Song Size:", len(data_txt))
        # process cfp
        for i, filename in enumerate(tqdm(data_txt)):  # i(key)= "train_data.txt" ,filename(value)=file name
            # file set
            cfp_file = os.path.join(self.cfp_dir, filename + ".npy")
            ref_file = os.path.join(self.f0_dir, filename + ".txt")
            # get raw cfp and freq
            temp_cfp = np.load(cfp_file, allow_pickle=True)  # temp_cfp:(3,360,time_frame)
            # temp_cfp[0, :, :] = temp_cfp[1, :, :] * temp_cfp[2, :, :]
            temp_freq = np.loadtxt(ref_file)  # temp_freq:(time_frame,2),label的time_frame less than temp_cfp的time_frame
            temp_freq = temp_freq[:, 1]  # per_frame f0,temp_freq:one dim
            # check length
            # synchronize between num_f0(label) with num_f0(cfp) based on  less num_f0(cfp or label)
            if temp_freq.shape[0] > temp_cfp.shape[2]:
                temp_freq = temp_freq[:temp_cfp.shape[2]]  # based on num_f0(cfp)
            else:
                temp_cfp = temp_cfp[:, :, :temp_freq.shape[0]]  # based on num_f0(label)
            # build per_data
            for j in range(0, temp_cfp.shape[2], shift_frame):
                # j=0,128,256...,last_frame
                bgnt = j  # per_segment stat_time
                endt = j + seg_frame  # per_segment end_time
                temp_x = temp_cfp[:, :, bgnt:endt]  # (3,360,128 or less than 128),last_seg may be less than 128 frames
                temp_gd = index2centf(temp_freq[bgnt:endt], self.cent_f)  # (128 or less than 128)
                # temp_gd is central frequency which is the most nearest and greater than per_frame f0(label)

                # padding 0 for last segment
                if temp_x.shape[2] < seg_frame:  # last segment may be less than 128 frames
                    rl = temp_x.shape[2]  # per_segment frame(128 or less than 128)
                    pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame))  # standard zero matrix:(3,360,128)
                    pad_gd = np.zeros((seg_frame))  # standard one dim zero matrix:(128)
                    pad_gd[:rl] = temp_gd
                    # the first rl frames in last segment have central frequency which is the most nearest
                    # and greater than per_frame f0(label),the rest frame in lat segment is 0 f0
                    pad_x[:, :, :rl] = temp_x  # same as above
                    temp_x = pad_x  # temp_x is processed cfp feature:(3,360,128)
                    temp_gd = pad_gd  # temp_gd is processed label:(128)
                temp_tx = reorganize(temp_x[:], config.octave_res)  # temp_tx(tcfp feature_map):(3,360,128)
                self.data_tcfp.append(temp_tx)  # integrate per_segment feature map into list
                self.data_cfp.append(temp_x)  # same as above
                self.data_gd.append(temp_gd)  # integrate per_segment label into list
        self.data_cfp = np.array(self.data_cfp)  # (11877, 3, 360, 128)
        # list[tensor(3,360,128),...,tensor(3,360,128)],gross_num_tensor=all train data_num_segment=11877
        # per_segment feature map:(3,360,128)
        self.data_tcfp = np.array(self.data_tcfp)  # same as above
        self.data_gd = np.array(self.data_gd)  # list[tensor(128),...,tensor(128)]：(11877,128)
        print("Total Datasize:", self.data_cfp.shape)  # (11877, 3, 360, 128)

    def __len__(self):
        return len(self.data_cfp)

    def __getitem__(self, index):  # create dict index for class(TONetTrainDataset)
        temp_dict = {
            "cfp": self.data_cfp[index].astype(np.float32),
            "tcfp": self.data_tcfp[index].astype(np.float32),
            "gd": self.data_gd[index]
        }
        return temp_dict


class TestDataset(Dataset):
    def __init__(self, data_list, config):
        self.config = config
        self.cfp_dir = os.path.join(config.data_path, config.cfp_dir)
        self.f0_dir = os.path.join(config.data_path, "f0ref")
        self.data_list = data_list
        self.cent_f = np.array(get_CenFreq(config.startfreq, config.stopfreq, config.octave_res))
        # init data array
        self.data_cfp = []
        self.data_gd = []
        self.data_len = []
        self.data_tcfp = []
        seg_frame = config.seg_frame
        shift_frame = config.shift_frame
        print("Data List:", data_list)
        with open(data_list, "r") as f:
            data_txt = f.readlines()
            data_txt = [d.split(".")[0] for d in data_txt]
        print("Song Size:", len(data_txt))
        # process cfp
        for i, filename in enumerate(tqdm(data_txt)):
            # i(key)= "data/test_adc.txt","data/test_mirex.txt","data/test_melody.txt" ,filename(value)=file name
            group_cfp = []
            group_gd = []
            group_tcfp = []
            # file set
            cfp_file = os.path.join(self.cfp_dir, filename + ".npy")
            ref_file = os.path.join(self.f0_dir, filename + ".txt")
            # get raw cfp and freq
            temp_cfp = np.load(cfp_file, allow_pickle=True)
            # temp_cfp[0, :, :] = temp_cfp[1, :, :] * temp_cfp[2, :, :]
            temp_freq = np.loadtxt(ref_file)
            temp_freq = temp_freq[:, 1]
            self.data_len.append(len(temp_freq))
            # check length
            if temp_freq.shape[0] > temp_cfp.shape[2]:
                temp_freq = temp_freq[:temp_cfp.shape[2]]
            else:
                temp_cfp = temp_cfp[:, :, :temp_freq.shape[0]]
            # build data
            for j in range(0, temp_cfp.shape[2], shift_frame):
                bgnt = j
                endt = j + seg_frame
                temp_x = temp_cfp[:, :, bgnt:endt]
                temp_gd = temp_freq[bgnt:endt]
                if temp_x.shape[2] < seg_frame:
                    rl = temp_x.shape[2]
                    pad_x = np.zeros((temp_x.shape[0], temp_x.shape[1], seg_frame))
                    pad_gd = np.zeros(seg_frame)
                    pad_gd[:rl] = temp_gd
                    pad_x[:, :, :rl] = temp_x
                    temp_x = pad_x  # temp_x:(3,360,128)
                    temp_gd = pad_gd  # temp_gd:(128)
                temp_tx = reorganize(temp_x[:], config.octave_res)  # temp_tx:(3,360,128)
                group_tcfp.append(temp_tx)  # one segment_feature map for three test_datasets
                group_cfp.append(temp_x)
                group_gd.append(temp_gd)
            group_tcfp = np.array(group_tcfp)  # all segment_feature map for three test_datasets
            group_cfp = np.array(group_cfp)
            group_gd = np.array(group_gd)
            # Why is it different from the training process?
            self.data_tcfp.append(group_tcfp)  # self.data_tcfp:(1,17/25/14,3,360,128)
            self.data_cfp.append(group_cfp)
            self.data_gd.append(group_gd)  # self.data_gd:(1,17/25/14,128)

    def __len__(self):
        return len(self.data_cfp)

    def __getitem__(self, index):  # create dict index for class(TONetTestDataset)
        temp_dict = {
            "cfp": self.data_cfp[index].astype(np.float32),
            "tcfp": self.data_tcfp[index].astype(np.float32),
            "gd": self.data_gd[index],
            "length": self.data_len[index]
        }
        return temp_dict
