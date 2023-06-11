"""

This file contains the main script

"""
import os
import random
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import config
from data_generator import TrainDataset, TestDataset
from model.msnet import MSnet
from model.tonet import TONet
from model.multi_dr import MLDRnet
from model.mtanet import MTAnet
from model.ftanet import FTAnet
from model.mcdnn import MCDNN
from model.danet import DAnet

from util import tonpy_fn

torch.set_num_threads(2)# define num_threads
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def train():
    train_dataset = TrainDataset(
        data_list=config.train_file,
        config=config
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        num_workers=config.n_workers,
        batch_size=config.batch_size
    )
    test_datasets = [
        TestDataset(
            data_list=d,
            config=config
        ) for d in config.test_file
    ]
    test_dataloaders = [
        DataLoader(
            dataset=d,
            shuffle=False,
            batch_size=1,
            collate_fn=tonpy_fn
        ) for d in test_datasets
    ]
    loss_func = nn.BCELoss()

    if config.model_type == "MCDNN":
        me_model = MCDNN()
        me_model_r = MCDNN()
    elif config.model_type == "MLDRNet":
        me_model = MLDRnet()
        me_model_r = MLDRnet()
    elif config.model_type == "FTANet":
        me_model = FTAnet(freq_bin=config.freq_bin, time_segment=config.seg_frame)
        me_model_r = FTAnet(freq_bin=config.freq_bin, time_segment=config.seg_frame)
    elif config.model_type == "MTAnet":
        me_model = MTAnet(input_channel = config.input_channel, drop_rate=0.1)
        me_model_r = MTAnet(input_channel = config.input_channel, drop_rate=0.1)
    else:  # MSNet
        me_model = MSnet()
        me_model_r = MSnet()
    if config.ablation_mode == "single":
        me_model_r = None
    model = TONet(
        l_model=me_model,
        r_model=me_model_r,
        config=config,
        loss_func=loss_func,
        mode=config.ablation_mode
    )
    trainer = pl.Trainer(
        deterministic=True,  # fix random seed
        # gpus=1,
        checkpoint_callback=False,
        max_epochs=config.max_epoch,
        auto_lr_find=True,
        sync_batchnorm=True,  # gpus used
        accelerator="gpu",
        devices=[6],
        # check_val_every_n_epoch = 1,
        val_check_interval=0.25,
        # fast_dev_run = True
        # num_sanity_val_steps=0
    )
    trainer.fit(model, train_dataloader, test_dataloaders)


def test():
    test_datasets = [
        TestDataset(
            data_list=d,
            config=config
        ) for d in config.test_file
    ]
    test_dataloaders = [
        DataLoader(
            dataset=d,
            shuffle=False,
            batch_size=1,
            collate_fn=tonpy_fn # integrate bs sets of data into one set of data
        ) for d in test_datasets
    ]
    loss_func = nn.BCELoss()

    if config.model_type == "MCDNN":
        me_model = MCDNN()
        me_model_r = MCDNN()
    elif config.model_type == "MLDRNet":
        me_model = MLDRnet()
        me_model_r = MLDRnet()
    elif config.model_type == "FTANet":
        me_model = FTAnet(freq_bin=config.freq_bin, time_segment=config.seg_frame)
        me_model_r = FTAnet(freq_bin=config.freq_bin, time_segment=config.seg_frame)
    elif config.model_type == "MTAnet":
        me_model = MTAnet(input_channel = config.input_channel, drop_rate=0.1)
        me_model_r = MTAnet(input_channel = config.input_channel, drop_rate=0.1)
    else:  # MSNet
        me_model = MSnet()
        me_model_r = MSnet()

    if config.ablation_mode == "single":
        me_model_r = None
    model = TONet(
        l_model=me_model,
        r_model=me_model_r,
        config=config,
        loss_func=loss_func,
        mode=config.ablation_mode
    )
    trainer = pl.Trainer(
        deterministic=True,
        # gpus=1,
        checkpoint_callback=False,
        max_epochs=config.max_epoch,
        auto_lr_find=True,
        sync_batchnorm=True,
        accelerator="gpu",
        devices=[3],
        # check_val_every_n_epoch = 1,
        val_check_interval=0.25,
    )
    # load the checkpoint
    ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    trainer.test(model, test_dataloaders)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MTANET for Singing Melody Extraction")
    subparsers = parser.add_subparsers(dest="mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    args = parser.parse_args()
    pl.seed_everything(config.random_seed)
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
train()
