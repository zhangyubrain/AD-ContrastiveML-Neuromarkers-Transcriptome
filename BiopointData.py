import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from construct_graph import read_data_regression, read_data_classification, read_data_regression_abOAS_f,\
    read_data_regression_hc, read_data_regression_hc, read_data_regression_hc_45, read_data_classification_hc_45, read_data_regression_pet_ab_adni, \
        read_data_regression_hc_CNdementiaContrast, regression_Longitudinal, regression_AbCN, regression_demo

class BiopointDataset(InMemoryDataset):
    def __init__(self, opt, name, transform=None, pre_transform=None):
        self.root = opt.dataroot
        self.opt = opt
        self.name = name
        self.task = opt.task
        super(BiopointDataset, self).__init__(opt.dataroot,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = self.root
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.

        if self.task == 'classification':
            self.data, self.slices = read_data_classification(self.opt)
        elif self.task == 'regression':
            # self.data, self.slices = read_data_regression_abOAS_f(self.opt)
            self.data, self.slices = read_data_regression(self.opt)
        elif self.task == 'regression_hc_visual':
            self.data, self.slices = read_data_regression_hc(self.opt)
        elif self.task == 'regression_hc_visual_CNdementiaContrast':
            self.data, self.slices = read_data_regression_hc_CNdementiaContrast(self.opt)
        elif self.task == 'regression_hc_visual_45':
            self.data, self.slices = read_data_regression_hc_45(self.opt)
        elif self.task == 'regression_ab_pet_adni':
            self.data, self.slices = read_data_regression_pet_ab_adni(self.opt)
        elif self.task == 'regression_Longitudinal':
            self.data, self.slices = regression_Longitudinal(self.opt)
        elif self.task == 'regression_A+CN':
            self.data, self.slices = regression_AbCN(self.opt)
        elif self.task == 'regression_demo':
            self.data, self.slices = regression_demo(self.opt)
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
