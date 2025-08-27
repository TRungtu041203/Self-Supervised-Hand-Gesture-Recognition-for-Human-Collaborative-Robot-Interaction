import sys
import argparse
import yaml
import math
import random
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class AimCLR_Processor(PT_Processor):
    """
        Processor for AimCLR Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2, data3], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            if self.arg.stream == 'joint':
                pass
            elif self.arg.stream == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)
                motion3 = torch.zeros_like(data3)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]
                motion3[:, :, :-1, :, :] = data3[:, :, 1:, :, :] - data3[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
                data3 = motion3
            elif self.arg.stream == 'bone':
                # COBOT bone connections: 42 hand joints (21 per hand) + 6 arm/shoulder joints
                Bone = [
                    # Arm and shoulder connections (6 joints: 43-48)
                    (43, 44), (44, 45), (45, 46), (46, 47), (47, 48),
                    # Right hand connections (21 joints: 1-21)
                    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
                    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
                    (17, 18), (18, 19), (19, 20), (20, 21),
                    # Left hand connections (21 joints: 22-42)
                    (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
                    (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
                    (38, 39), (39, 40), (40, 41), (41, 42),
                    # Connect hands to arms
                    (21, 43), (42, 48),  # Connect right hand to right arm, left hand to left arm
                ]

                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)
                bone3 = torch.zeros_like(data3)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                    bone3[:, :, :, v1 - 1, :] = data3[:, :, :, v1 - 1, :] - data3[:, :, :, v2 - 1, :]

                data1 = bone1
                data2 = bone2
                data3 = bone3
            else:
                raise ValueError

            # forward
            if epoch <= self.arg.mining_epoch:
                output1, target1, output2, target2 = self.model(data1, data2, data3)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = self.loss(output1, target1)
                loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                loss = loss1 + loss2
            else:
                output1, mask, output2, target2 = self.model(data1, data2, data3, nnm=True, topk=self.arg.topk)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = - (F.log_softmax(output1, dim=1) * mask).sum(1) / mask.sum(1)
                loss1 = loss1.mean()
                loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                loss = loss1 + loss2

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        return parser
