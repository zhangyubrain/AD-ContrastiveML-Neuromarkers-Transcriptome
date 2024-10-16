
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:11:42 2023

@author: 99488
"""


import os
from os import listdir
import numpy as np
import random
import argparse
import time
import copy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import torch
from torch.optim import lr_scheduler
import sys
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append('/home/alex/project/CGCN/utils')
sys.path.append('/home/alex/project/CGCN')
sys.path.append(r'/home/alex/project/utils_for_all')
from BiopointData import BiopointDataset
from torch_geometric.data import DataLoader
from net.FocalLoss import focal_loss
from net.brain_networks_ad import ContrativeNet_infomax
from tkinter import _flatten
import scipy.io as sio
from util import (normal_transform_train,normal_transform_test,train_val_test_split, sens_spec, site_split,get_index,plot_ROC,
                  write_excel_xlsx, plot_fea, plot_distr)
from mmd_loss import MMD_loss
from sklearn.model_selection import KFold, StratifiedKFold
from torch.nn import MSELoss, KLDivLoss, CrossEntropyLoss
from common_utils import keep_triangle_half, heatmap, vector_to_matrix
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from networkx.convert_matrix import from_numpy_matrix
import networkx as nx
from sklearn.metrics import r2_score, mean_squared_error
import torch.nn as nn
import torch.nn.functional as F

def ContrastiveLoss(data1, data2, y_pred_dis, 
       y_pred_dis_share, y_pred_hc, logits_dis, logits_dis_share, logits_hc, opt):

    pos_loss = -torch.log(logits_dis).mean()
    neg_loss_dis_share = -torch.log(1 -logits_dis_share).mean()
    neg_loss_hc = -torch.log(1 -logits_hc).mean()

    contrastive_loss = pos_loss + neg_loss_dis_share + neg_loss_hc

    pos_loss_hc = -torch.log(logits_hc).mean()
    pos_loss_dis_share = -torch.log(logits_dis_share).mean()
    neg_loss_dis = -torch.log(1 -logits_dis).mean()

    contrastive_loss2 = pos_loss_hc + pos_loss_dis_share + neg_loss_dis
    
    predict_hc_loss = torch.sqrt(MSELoss()(y_pred_hc[~data1.pcd[:,3].isnan()].squeeze(), 
                            (data1.pcd[~data1.pcd[:,3].isnan(),3]).squeeze()))
    predict_dis_loss = torch.sqrt(MSELoss()(y_pred_dis[~data2.pcd[:,3].isnan()].squeeze(), 
                                      (data2.pcd[~data2.pcd[:,3].isnan(),3]).squeeze()))
    predict_loss = opt.theta1 * predict_dis_loss + opt.theta2 * predict_hc_loss #+ predict_dis_share_loss 
    
    loss_all = predict_loss + opt.alpha*(contrastive_loss+contrastive_loss2)
    
    return loss_all, opt.alpha*contrastive_loss, predict_loss, predict_dis_loss

def ContrastiveLoss_pretrain(data1, data2, logits_dis, logits_dis_share, logits_hc, opt):

    pos_loss = -torch.log(logits_dis).mean()
    neg_loss_dis_share = -torch.log(1 -logits_dis_share).mean()
    neg_loss_hc = -torch.log(1 -logits_hc).mean()

    contrastive_loss = pos_loss + neg_loss_dis_share + neg_loss_hc
    
    return opt.alpha*contrastive_loss

def ContrastiveLoss_pretrain2(data1, data2, logits_dis, logits_dis_share, logits_hc, opt):

    pos_loss_hc = -torch.log(logits_hc).mean()
    pos_loss_dis_share = -torch.log(logits_dis_share).mean()
    neg_loss_dis = -torch.log(1 -logits_dis).mean()

    contrastive_loss = pos_loss_hc + pos_loss_dis_share + neg_loss_dis
    
    return opt.alpha*contrastive_loss

def dist_loss(s,ratio):
    s = s.sort(dim=1).values
    source = s[:,-int(s.size(1)*ratio):]
    target = s[:,:int(s.size(1)*ratio)]
    res =  MMD_loss()(source,target)
    return -res

def dist_loss2(source,target):
    res =  MMD_loss()(source,target)
    return -res
    
def main():
    
    def train(hc_tr_loader, patient_tr_loader, pretrain):
        print('train...........')
        model.train()

        loss_all = 0
        contras_loss_all = 0
        predict_loss_all = 0
        y_dis_pred_all = []
        y_hc_pred_all = []
        y_dis_true_all = []
        y_hc_true_all = []

        i = 0
        for data1, data2 in zip(hc_tr_loader, patient_tr_loader):

            data1 = data1.to(device)
            data2 = data2.to(device)
            optimizer.zero_grad()
            
            output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
              data2.x, data2.edge_index, data2.edge_attr, data1.pcd, data2.eyes, data2.batch)
            
            if pretrain:
                loss = ContrastiveLoss_pretrain(data1, data2, 
                              logits_dis, logits_dis_share, logits_hc, opt)  
                loss2 = ContrastiveLoss_pretrain2(data1, data2, 
                              logits_dis, logits_dis_share, logits_hc, opt)  
                contras_loss = loss + loss2
                predict_loss = 0
                predict_dis_loss = 0
                contras_loss_all += contras_loss.item() 
                predict_loss_all += predict_loss
            else:
                loss, contras_loss, predict_loss, predict_dis_loss = ContrastiveLoss(data1, data2, y_pred_dis, 
                       y_pred_dis_share, y_pred_hc, logits_dis, logits_dis_share, logits_hc, opt)            # raw_x = output.view(60,100,100)
                contras_loss_all += contras_loss.item() 
                predict_loss_all += predict_loss.item() 
            y_dis_pred_all.extend(y_pred_dis[~data2.pcd[:,3].isnan()].detach().cpu().tolist())
            y_hc_pred_all.extend(y_pred_hc[~data1.pcd[:,3].isnan()].detach().cpu().tolist())
            y_hc_true_all.extend(data1.pcd[~data1.pcd[:,3].isnan(),3].detach().cpu().tolist())
            y_dis_true_all.extend(data2.pcd[~data2.pcd[:,3].isnan(),3].detach().cpu().tolist())
            i = i + 1
    
            loss.backward()
            loss_all += loss.item() 
            # optimzercenter.step()
            optimizer.step()
            scheduler.step()

        y_dis_pred_all = np.array(y_dis_pred_all)
        y_hc_pred_all = np.array(y_hc_pred_all)
        y_dis_true_all = np.array(y_dis_true_all)
        y_hc_true_all = np.array(y_hc_true_all)
        
        r2_hc = r2_score(y_hc_true_all, (y_hc_pred_all))
        r2_dis = r2_score(y_dis_true_all, (y_dis_pred_all))
        r2_all = r2_score(np.r_[y_hc_true_all, y_dis_true_all], 
                          np.r_[(y_hc_pred_all), (y_dis_pred_all)])
        
        r_hc, _ = pearsonr(y_hc_true_all, (y_hc_pred_all).squeeze())
        r_dis, _ = pearsonr(y_dis_true_all, (y_dis_pred_all).squeeze())
        r_all, _ = pearsonr(np.r_[y_hc_true_all, y_dis_true_all], 
                          np.r_[(y_hc_pred_all).squeeze(), (y_dis_pred_all).squeeze()])
        
        return loss_all / i, contras_loss_all/i, \
            predict_loss_all / i, r2_hc, r2_dis, r2_all, r_hc, r_dis, r_all

    def test_loss(hc_te_loader, patient_te_loader):
        print('testing...........')
        model.eval()
        loss_all = 0
        contras_loss_all = 0
        predict_loss_all = 0
        predict_dis_loss_all = 0
        y_dis_pred_all = []
        y_hc_pred_all = []
        y_dis_true_all = []
        y_hc_true_all = []

        i=0
        for data1, data2 in zip(hc_te_loader, patient_te_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)

            output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
              data2.x, data2.edge_index, data2.edge_attr, data1.pcd, data2.eyes, data2.batch)
                    
            loss, contras_loss, predict_loss, predict_dis_loss = ContrastiveLoss(data1, data2, y_pred_dis, 
                   y_pred_dis_share, y_pred_hc, logits_dis, logits_dis_share, logits_hc, opt)              # raw_x = output.view(60,100,100)
                
            y_dis_pred_all.extend(y_pred_dis[~data2.pcd[:,3].isnan()].detach().cpu().tolist())
            y_hc_pred_all.extend(y_pred_hc[~data1.pcd[:,3].isnan()].detach().cpu().tolist())
            y_hc_true_all.extend(data1.pcd[~data1.pcd[:,3].isnan(),3].detach().cpu().tolist())
            y_dis_true_all.extend(data2.pcd[~data2.pcd[:,3].isnan(),3].detach().cpu().tolist())

            # loss2 = distance(edge_raw, output)
            # loss2 = 0
            # loss_edge = distance(data.x, edge)

            # kl_loss = torch.mean(torch.exp(log_var) + mu**2 - 1.0 - log_var)
            # kl_loss_edge = torch.mean(torch.exp(log_var_e) + mu_e**2 - 1.0 - log_var_e)
            # loss = loss_edge

            # loss = loss + kl_loss*0.5 
            i = i + 1

            loss_all += loss.item() 
            contras_loss_all += contras_loss.item() 
            predict_loss_all += predict_loss.item() 
            predict_dis_loss_all += predict_dis_loss.item() 
            
        y_dis_pred_all = np.array(y_dis_pred_all)
        y_hc_pred_all = np.array(y_hc_pred_all)
        y_dis_true_all = np.array(y_dis_true_all)
        y_hc_true_all = np.array(y_hc_true_all)
        
        r2_hc = r2_score(y_hc_true_all, (y_hc_pred_all))
        r2_dis = r2_score(y_dis_true_all, (y_dis_pred_all))
        r2_all = r2_score(np.r_[y_hc_true_all, y_dis_true_all], 
                          np.r_[(y_hc_pred_all), (y_dis_pred_all)])
        
        r_hc, _ = pearsonr(y_hc_true_all, (y_hc_pred_all).squeeze())
        r_dis, _ = pearsonr(y_dis_true_all, (y_dis_pred_all).squeeze())
        r_all, _ = pearsonr(np.r_[y_hc_true_all, y_dis_true_all], 
                          np.r_[(y_hc_pred_all).squeeze(), (y_dis_pred_all).squeeze()])

        
        return loss_all / i, contras_loss_all/i, \
            predict_loss_all / i, predict_dis_loss_all / i, r2_hc, r2_dis, r2_all, r_hc, r_dis, r_all
    
    
    def predict(hc_te_loader, patient_te_loader):
        y_dis_pred_all = []
        y_hc_pred_all = []
        y_dis_true_all = []
        y_hc_true_all = []
        y_hc_mask_all = []
        y_ad_mask_all = []
        model.eval()
        for data1, data2 in zip(hc_te_loader, patient_te_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
              data2.x, data2.edge_index, data2.edge_attr, data1.pcd, data2.eyes, data2.batch)
                    
            y_dis_pred_all.extend(y_pred_dis[~data2.pcd[:,3].isnan()].detach().cpu().tolist())
            # y_dis_pred_all.extend(y_pred_dis_share[~data2.pcd[:,3].isnan()].detach().cpu().tolist())
            y_hc_pred_all.extend(y_pred_hc[~data1.pcd[:,3].isnan()].detach().cpu().tolist())
            y_hc_true_all.extend(data1.pcd[~data1.pcd[:,3].isnan(),3].detach().cpu().tolist())
            y_dis_true_all.extend(data2.pcd[~data2.pcd[:,3].isnan(),3].detach().cpu().tolist())
            y_hc_mask_all.extend(data1.pcd[:,3].isnan().detach().cpu().tolist())
            y_ad_mask_all.extend(data2.pcd[:,3].isnan().detach().cpu().tolist())

        y_dis_pred_all = (np.array(y_dis_pred_all))
        y_hc_pred_all = (np.array(y_hc_pred_all))
        # y_dis_pred_all = np.array(y_dis_pred_all)
        # y_hc_pred_all = np.array(y_hc_pred_all)
        y_dis_true_all = np.array(y_dis_true_all)
        y_hc_true_all = np.array(y_hc_true_all)
        y_true = np.r_[y_hc_true_all, y_dis_true_all]
        y_predict = np.r_[y_hc_pred_all, y_dis_pred_all]
        
        return y_dis_pred_all.tolist(), y_hc_pred_all.tolist(), y_dis_true_all.tolist(),\
            y_hc_true_all.tolist(), y_true.tolist(), y_predict.tolist(), y_hc_mask_all, y_ad_mask_all

                
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    EPS = 1e-15
    device = torch.device("cuda:0")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', type=str, default=\
                        r'xxxx\', 
                        help='result save path')
    parser.add_argument('--model_file', type=str, default=\
                        r'xxxx\', 
                        help='model save path')
    parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default= 100, help='size of the batches')
    parser.add_argument('--fold', type=int, default=5, help='training which fold')
    parser.add_argument('--lr', type = float, default=0.1, help='learning rate')
    parser.add_argument('--stepsize', type=int, default=250, help='scheduler step size')
    # parser.add_argument('--stepsize', type=int, default=22, help='scheduler step size')
    parser.add_argument('--weightdecay', type=float, default=0.01, help='regularization')
    # parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
    parser.add_argument('--gamma', type=float, default=0.4, help='scheduler shrinking rate')
    parser.add_argument('--alpha', type=float, default=1, help='loss control to disentangle HC and disorder')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
    parser.add_argument('--beta', type=float, default=1, help='loss control to force gaussian distribution')
    parser.add_argument('--theta1', type=float, default=1, help='loss control to prediction task for dis')
    parser.add_argument('--theta2', type=float, default=0.2, help='loss control to prediction task for HC')
    parser.add_argument('--build_net', default=True, type=bool, help='model name')
    parser.add_argument('--in_channels', type=int, default=100)
    parser.add_argument('--hidden_channels', type=int, default=50)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--conv', type=str, default='edgeconv', help='edgeconv || gat || gcn || gen')
    parser.add_argument('--act', type=str, default='tanh', help='relu || leaky_relu || prelu || tanh')
    parser.add_argument('--sum_res', type=bool, default=True)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--normalization', action='store_true') 
    parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--dataroot', type=str,
                        default=r'/home/alex/project/CGCN/A+/ab/data/',
                        help='root directory of the dataset')
    parser.add_argument('--retrain', default=False, type=bool, help='whether train from used model')     
    parser.add_argument('--epsilon', default=0.1, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
    parser.add_argument('--demean', type=bool, default=True)
    parser.add_argument('--drop', default=0.3, type=float, help='drop ratio')
    parser.add_argument('--task', default='regression_hc_visual', type=str, help='classfication / regression/regression_hc_visual/classification_hc_visual')
    parser.add_argument('--augmentation', default=5, type=int, help='times of augmentation')
    parser.add_argument('--cluster', default=7, type=int, help='cluster number')

    parser.set_defaults(save_model=True)
    parser.set_defaults(normalization=True)
    opt = parser.parse_args()
    name = 'Biopoint'

    ###########################15% edge keep as graph, HC aug used, only with ab, ptau are kept, test ensemble-nonensemble 
    ############# Define Dataloader -- need costumize#####################
    dataset = BiopointDataset(opt, name)
    HC_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0]
    ad_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0]
    hc_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),3][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0].numpy()
    hc_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]==0,3].numpy())
    hc_target_idx = np.where(~np.isnan(hc_target))[0]
    ad_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),3][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
    ad_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]!=0,3].numpy())
    ad_target_idx = np.where(~np.isnan(ad_target))[0]
    HC_data_aug = dataset[dataset.data.pcd[:,-1]==0]
    ad_data_aug = dataset[dataset.data.pcd[:,-1]!=0]
    # sio.savemat(r'/home/alex/project/CGCN/A+/PCD.mat', {pcd: })

    for t in range(10):
        t = t+10
        if torch.cuda.is_available():
            # setup_seed(666) 
            setup_seed(t) 
        opt.save_file = r'result/seed2/{}/{}/update_contrast_loss'.format(opt.theta2, t)
        opt.model_file = r'model/seed2/{}/{}/update_contrast_loss'.format(opt.theta2, t)
            
        if not os.path.exists(opt.save_file):
            os.makedirs(opt.save_file)

        kf = KFold(n_splits = opt.fold, shuffle=True)
        i = 0
        all_predict_hc = []
        all_true_hc = []
        all_predict_dis = []
        all_true_dis = []
        all_predict = []
        all_true = []
        all_r_tr = []
        if opt.fold:
           ########all cv
            for index1, index2 in zip(kf.split(hc_target_idx), kf.split(ad_target_idx)):
                setup_seed(2) 
                i = i + 1
                tr_index1, val_index1 = index1
                tr_index2, val_index2 = index2
                train_mask_hc = torch.zeros(len(hc_target)*(opt.augmentation+1), dtype=torch.bool)
                test_mask_hc = torch.zeros(len(hc_target), dtype=torch.bool)
                train_mask_pati = torch.zeros(len(ad_data), dtype=torch.bool)
                # train_mask_pati = torch.zeros(len(ad_data)*(opt.augmentation+1), dtype=torch.bool)
                test_mask_pati = torch.zeros(len(ad_data), dtype=torch.bool)

                aug_tr_idx_hc = []
                aug_tr_idx_dem = []
                for j in range(opt.augmentation+1):
                    aug_tr_idx_hc.extend(hc_target_idx[tr_index1]+j*len(hc_target))
                    aug_tr_idx_dem.extend(ad_target_idx[tr_index2]+len(ad_target)*j)
                aug_tr_idx_hc = np.array(aug_tr_idx_hc)
                aug_tr_idx_dem = np.array(aug_tr_idx_dem)
                train_mask_hc[aug_tr_idx_hc] = True
                train_mask_hc[hc_notarget_mask] = True
                test_mask_hc[hc_target_idx[val_index1]] = True
                train_mask_pati[ad_target_idx[tr_index2]] = True
                train_mask_pati[np.isnan(ad_target)] = True
                # train_mask_pati[aug_tr_idx_dem] = True
                test_mask_pati[ad_target_idx[val_index2]] = True
                
                test_hc = copy.deepcopy(HC_data[test_mask_hc])
                train_hc = copy.deepcopy(HC_data_aug[train_mask_hc])
                test_pati = copy.deepcopy(ad_data[test_mask_pati])
                # train_pati = copy.deepcopy(ad_data_aug[train_mask_pati])
                train_pati = copy.deepcopy(ad_data[train_mask_pati])
                 # ###################### Normalize features ##########################
                if opt.normalization:
                    for j in range(train_hc.data.x.shape[1]):
                        train_hc.data.x[:, j], lamb, xmean, xstd = normal_transform_train(train_hc.data.x[:, j])
                        test_hc.data.x[:, j] = normal_transform_test(test_hc.data.x[:, j],lamb, xmean, xstd)
                    for j in range(train_hc.data.x.shape[1]):
                        train_pati.data.x[:, j], lamb, xmean, xstd = normal_transform_train(train_pati.data.x[:, j])
                        test_pati.data.x[:, j] = normal_transform_test(test_pati.data.x[:, j],lamb, xmean, xstd)
    
                    ############### Define Graph Deep Learning Network ##########################
                if opt.build_net:

                    model = ContrativeNet_infomax(opt).to(device)

                if opt.retrain:

                    checkpoint  = torch.load(opt.model_file, map_location=torch.device("cuda:0"))

                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
            
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
            
                print(model)
                ##############################################################           
        
                if opt.optimizer == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
                elif opt.optimizer == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)
                
                scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
                            
                if opt.retrain:
                    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
                    scheduler.load_state_dict(checkpoint['scheduler'])  # 加载优化器参数
                    # start_epoch = checkpoint['epoch']  # 设置开始的epoch
            
                #######################################################################################
                ############################   Model Training #########################################
                #######################################################################################
                # best_model_wts = copy.deepcopy(model.state_dict())
                best_loss = -10
                all_result=list()
            
                all_predict_ = []
                all_true_ = []
                all_predict_dis_ = []
                all_true_dis_ = []
                all_predict_hc_ = []
                all_true_hc_ = []
                all_r_tr_ = []
                for epoch in range(0, opt.n_epochs):
                    ####################more NC
                    idx1 = np.random.randint(0, len(train_hc), size=(len(train_pati), 1)).squeeze()
                    idx2 = np.random.randint(0, len(train_hc), size=(len(test_pati), 1)).squeeze()
    
                    patient_tr_loader = DataLoader(train_pati,batch_size=opt.batchSize,shuffle = True)
                    hc_tr_loader = DataLoader(train_hc.index_select(list(idx1)),batch_size=opt.batchSize, shuffle= True)
                    hc_te_loader = DataLoader(train_hc.index_select(list(idx2)),batch_size=opt.batchSize, shuffle= True)
    
                    patient_te_loader = DataLoader(test_pati,batch_size=opt.batchSize, shuffle= True)
                    
                    since  = time.time()
                    if epoch < 15:
                        loss_tr, contras_loss_tr, predict_loss_tr,\
                            r2_hc_tr, r2_dis_tr, r2_all_tr, r_hc_tr, r_dis_tr, r_all_tr= train(hc_tr_loader, patient_tr_loader, True)
                    else:
                        loss_tr, contras_loss_tr, predict_loss_tr,\
                            r2_hc_tr, r2_dis_tr, r2_all_tr, r_hc_tr, r_dis_tr, r_all_tr= train(hc_tr_loader, patient_tr_loader, False)
                 
                    loss_val, contras_loss_val, predict_loss_val, predict_dis_loss_val, \
                        r2_hc_te, r2_dis_te, r2_all_te, r_hc_te, r_dis_te, r_all_te = test_loss(hc_te_loader, patient_te_loader)
                    time_elapsed = time.time() - since
                    # test_accuracy, prediction,label,predict_labels = test_acc(test_loader)
                    print('*====**')
                    print('{:.0f}m {:.5f}s'.format(time_elapsed / 60, time_elapsed % 60))
                    print('''Epoch: {:03d}, Train Loss all: {:.7f}, Prediction training loss: {:.7f}, 
                          r2 of HC training: {:.7f}, r2 of dis training: {:.7f}, r2 of all training: {:.7f},
                          r of HC training: {:.7f}, r of dis training: {:.7f}, r of all training: {:.7f},
                          Val Loss all: {:.7f}, Prediction val loss: {:.7f},r2 of HC test: {:.7f},
                          r2 of dis test: {:.7f}, r2 of all test: {:.7f}, r of HC test: {:.7f},
                          r of dis test: {:.7f}, r of all test: {:.7f}'''.format(epoch, 
                               loss_tr, predict_loss_tr, r2_hc_tr, r2_dis_tr, r2_all_tr, r_hc_tr, r_dis_tr, r_all_tr,
                               loss_val, predict_loss_val, r2_hc_te, r2_dis_te, r2_all_te, r_hc_te, r_dis_te, r_all_te))
                    if epoch > 90:
                        best_loss = r2_dis_te
                        print("saving best model")
                        checkpoint = {
                            "net": model.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'scheduler':scheduler.state_dict(),
                            "epoch": epoch
                        }
                        if not os.path.exists(opt.model_file):
                            os.makedirs(opt.model_file)
                        if opt.save_model:
                            torch.save(checkpoint, os.path.join(opt.model_file,'model_cv_{}.pth'.format(i)))
                            
                        y_dis_pred_all, y_hc_pred_all, y_dis_true_all, y_hc_true_all, y_true, y_predict, \
                            _, _ = predict(hc_te_loader, patient_te_loader)
                        all_predict_dis_.append(y_dis_pred_all)
                        all_true_dis_.append(y_dis_true_all)
                        all_predict_hc_.append(y_hc_pred_all)
                        all_true_hc_.append(y_hc_true_all)
                        all_predict_.append(y_predict)
                        all_true_.append(y_true)
                        all_r_tr_.append([r2_hc_tr, r2_dis_tr, r2_all_tr, r_hc_tr, r_dis_tr, r_all_tr])
                        
                    all_result.append([loss_tr, contras_loss_tr, predict_loss_tr,
                                       r2_hc_tr, r2_dis_tr, r2_all_tr, r_hc_tr, r_dis_tr, r_all_tr,
                                       loss_val, contras_loss_val, predict_loss_val,
                                       r2_hc_te, r2_dis_te, r2_all_te, r_hc_te, r_dis_te, r_all_te])
                    # te_true_hc.append(y_hc_true_te)
                    # te_true_dis.append(te_true_dis)
    
                y_dis_pred_tr, y_hc_pred_tr, y_dis_true_tr, y_hc_true_tr, _, _, \
                    _, _ = predict(hc_tr_loader, patient_tr_loader)

                # all_predict_dis.extend(y_dis_pred_all)
                # all_true_dis.extend(y_dis_true_all)
                # all_r_tr.append([r2_hc_tr, r2_dis_tr, r2_all_tr, r_hc_tr, r_dis_tr, r_all_tr])
                all_predict_hc.extend(all_predict_hc_[-1])
                all_true_hc.extend(all_true_hc_[-1])
                all_predict_dis.extend(all_predict_dis_[-1])
                all_true_dis.extend(all_true_dis_[-1])
                all_predict.extend(all_predict_[-1])
                all_true.extend(all_true_[-1])
                all_r_tr.append(all_r_tr_[-1])
                all_result = np.array(all_result)
                sio.savemat(os.path.join(opt.save_file, "{}.mat".format(i)),{'all_loss': all_result, 'te_true_dis': y_dis_true_all,
                                                                             'te_pred_dis': y_dis_pred_all, 'tr_pred_dis': y_dis_pred_tr, 
                                                                             'te_pred_hc': y_hc_pred_all, 'tr_pred_hc': y_hc_pred_tr, 
                                                                             'tr_true_dis': y_dis_true_tr, 'tr_true_hc': y_hc_true_tr,
                                                                             # 'ad_index': val_index2[idx2],
                                                                             'idx_tr_sampling': idx1,'te_true_hc': y_hc_true_all,
                                                                             'idx_te_sampling': idx2, 'stop_r2': best_loss})
                
                def plot_loss(tr_loss, te_loss, save_file, name, i):
                    plt.figure(figsize =(15,15))
                    plt.plot(np.arange(len(tr_loss))+1, tr_loss, color = 'r', label='training set')
                    plt.plot(np.arange(len(tr_loss))+1, te_loss, color = 'b', label='test set')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend(loc='best') 
                    plt.savefig(os.path.join(save_file,"{}_{}.svg".format(name, i)), bbox_inches = 'tight') 
                    # plt.show()
                    plt.close()
                    
                plot_loss(all_result[:,0], all_result[:,9], opt.save_file, 'loss_all_cv', i)
                plot_loss(all_result[:,1], all_result[:,10], opt.save_file, 'loss_contrastive_cv', i)
                plot_loss(all_result[:,2], all_result[:,11], opt.save_file, 'loss_predict_cv', i)
                plot_loss(all_result[:,3], all_result[:,12], opt.save_file, 'r2_hc_cv', i)
                plot_loss(all_result[:,4], all_result[:,13], opt.save_file, 'r2_dis_cv', i)
                plot_loss(all_result[:,6], all_result[:,15], opt.save_file, 'r_hc_cv', i)
                plot_loss(all_result[:,8], all_result[:,17], opt.save_file, 'r_all_cv', i)
            # mse = mean_squared_error(all_true, all_predict)
            # pearsonr_all = []
            # for i in range(len(all_true)):
            #     r = pearsonr(all_true[i], all_predict[i])
            #     pearsonr_all.append(r)
            # pearsonr_mean = np.mean(abs(np.array(pearsonr_all))[:,0])
            all_predict_hc = np.array(all_predict_hc).squeeze()
            all_true_hc = np.array(all_true_hc).squeeze()
            all_predict_dis = np.array(all_predict_dis).squeeze()
            all_true_dis = np.array(all_true_dis).squeeze()
            all_predict = np.array(all_predict).squeeze()
            all_true = np.array(all_true).squeeze()
            
            r2_hc = r2_score(all_true_hc, all_predict_hc)
            r2_dis = r2_score(all_true_dis, all_predict_dis)
            r2_all = r2_score(all_true, all_predict)
            
            r_hc,_ = pearsonr(all_true_hc, all_predict_hc)
            r_dis,_ = pearsonr(all_true_dis, all_predict_dis)
            r_all,_ = pearsonr(all_true, all_predict)
            
            sio.savemat(os.path.join(opt.save_file, "predict_outcome.mat"),{'all_predict_hc': all_predict_hc,
                        'all_true_hc': all_true_hc, 'all_predict_dis': all_predict_dis, 'all_true_dis':
                            all_true_dis, 'all_predict': all_predict, 'all_true': all_true, 
                            'r2_dis': r2_dis, 'r_dis': r_dis, 'all_r_tr': all_r_tr, 'r2_hc': r2_hc, 'r_hc': r_hc,
                            'r2_all': r2_all, 'r_all': r_all})
    
            
            f = open(os.path.join(opt.model_file,'ACC_SPE_SEN_each_epoch.txt'),'a')
            f.write('\n model:{}'.format(str(model)))
            f.write('\n parameter:{}'.format(str(opt)))
            # f.write('\n r2_hc:{}, r2_dis:{}, r2_all:{}, r_hc:{}, r_dis:{}, r_all:{}'.format(
            #     r2_hc, r2_dis, r2_all, r_hc, r_all))
            # f.write('\n pearsonr_mean:{}'.format(pearsonr_mean))
            f.close()
            # break

    return


if __name__ == '__main__':
    main()    
