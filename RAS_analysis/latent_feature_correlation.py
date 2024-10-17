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
# import shap
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer

import matplotlib.pyplot as plt
import torch
from torch.optim import lr_scheduler
import sys
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append('/home/alex/project/CGCN/utils')
sys.path.append('/home/alex/project/CGCN')
sys.path.append(r'/home/alex/project/utils_for_all')

from BiopointData import BiopointDataset
from torch_geometric.data import DataLoader
from net.FocalLoss import focal_loss
from net.brain_networks_ad import GAE2, GAE3, ContrativeNet_infomax
# from net.brain_networks_ad_ablation_ab_no_contrast import ContrativeNet_infomax
# from net.brain_networks_ad_ablation_ab import ContrativeNet_infomax
from tkinter import _flatten
import scipy.io as sio
from util import (normal_transform_train,normal_transform_test,train_val_test_split, sens_spec, site_split,get_index,plot_ROC,
                  write_excel_xlsx, plot_fea, plot_distr)
from mmd_loss import MMD_loss
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.nn import GraphUNet
from torch.nn import MSELoss, KLDivLoss, CrossEntropyLoss
from common_utils import keep_triangle_half, heatmap, vector_to_matrix, setup_seed
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, f_oneway, spearmanr
from statsmodels.stats import multitest
from networkx.convert_matrix import from_numpy_matrix
import networkx as nx
from sklearn.metrics import r2_score, mean_squared_error
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_score
from matplotlib.pyplot import MultipleLocator
from sklearn.decomposition import PCA
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
from mmd_loss import MMD_loss
from scipy.special import kl_div
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse
from scipy.spatial.distance import cdist, pdist
from torch_geometric.explain import Explainer, PGExplainer, CaptumExplainer
from skbio.stats.distance import mantel
from scipy import stats

font = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 30}

def ContrastiveLoss_tr(data1, data2, out_dis_hc, out_hc, out_dis, mu_hc, log_var_hc, mu_dis_hc, 
                    log_var_dis_hc, mu_dis, log_var_dis, z_hc, z_dis_hc, z_dis, y_pred_dis, 
                    y_pred_dis_share, y_pred_hc, y_prob_dis, y_prob_dis_share, opt):

    #reconstruct loss
    reconstruction_loss_hc = MSELoss()(data1.x, out_hc)
    reconstruction_loss_dis_share = MSELoss()(data2.x, out_dis_hc)
    reconstruction_loss_dis = MSELoss()(data2.x, out_dis)
    reconstruction_loss = reconstruction_loss_hc + reconstruction_loss_dis_share + reconstruction_loss_dis
    # reconstruction_loss = reconstruction_loss_hc + reconstruction_loss_dis
    
    # loss for [1,0] normal distribution, useing mean instead of sum to reduce the scaler effect
    kl_loss_hc = torch.mean(torch.exp(log_var_hc) + mu_hc**2 - 1.0 - log_var_hc)
    kl_loss_dis_share = torch.mean(torch.exp(log_var_dis_hc) + mu_dis_hc**2 - 1.0 - log_var_dis_hc)
    kl_loss_dis = torch.mean(torch.exp(log_var_dis) + mu_dis**2 - 1.0 - log_var_dis)
    kl_loss = kl_loss_hc + kl_loss_dis_share + kl_loss_dis

    fea = torch.cat((y_prob_dis, y_prob_dis_share), 0).type(torch.float)
    label = torch.zeros((fea.shape[0])).type(torch.float) 
    label[:len(z_dis)] = 1.0
    device = torch.device("cuda:0")
    label = label.to(device)

    distengle_loss1 = F.binary_cross_entropy(fea.squeeze(), label)
    del label
    contrastive_loss = distengle_loss1


    # contrastive_loss = similar_loss + distengle_loss1 + distengle_loss2


        # regression loss
        # predict_hc_loss = MSELoss()(y_pred_hc[~data1.pcd[:,3].isnan()].squeeze(), 
        #                         torch.log(data1.pcd[~data1.pcd[:,3].isnan(),3]).squeeze())
        # predict_dis_loss = MSELoss()(y_pred_dis[~data2.pcd[:,3].isnan()].squeeze(), 
        #                                   torch.log(data2.pcd[~data2.pcd[:,3].isnan(),3]).squeeze())
    # predict_hc_loss = nn.L1Loss()(y_pred_hc[~data1.pcd[:,3].isnan()].squeeze(), 
    #                         torch.log(data1.pcd[~data1.pcd[:,3].isnan(),3]).squeeze())
    # predict_dis_loss = nn.L1Loss()(y_pred_dis[~data2.pcd[:,3].isnan()].squeeze(), 
    #                                   torch.log(data2.pcd[~data2.pcd[:,3].isnan(),3]).squeeze())
    predict_hc_loss = torch.sqrt(MSELoss()(y_pred_hc[~data1.pcd[:,3].isnan()].squeeze(), 
                            torch.log(data1.pcd[~data1.pcd[:,3].isnan(),3]).squeeze()))
    predict_dis_loss = torch.sqrt(MSELoss()(y_pred_dis[~data2.pcd[:,3].isnan()].squeeze(), 
                                      torch.log(data2.pcd[~data2.pcd[:,3].isnan(),3]).squeeze()))
    predict_loss = predict_dis_loss + predict_hc_loss 
    
    loss_all = reconstruction_loss + opt.beta*kl_loss + opt.alpha*contrastive_loss  + opt.theta*predict_loss 
    
    return loss_all, reconstruction_loss, opt.beta*kl_loss, opt.alpha*contrastive_loss, predict_loss
    

def ContrastiveLoss_te(data1, data2, out_dis_hc, out_hc, out_dis, mu_hc, log_var_hc, mu_dis_hc, 
                    log_var_dis_hc, mu_dis, log_var_dis, z_hc, z_dis_hc, z_dis, y_pred_dis, 
                    y_pred_dis_share, y_pred_hc, y_prob_dis, y_prob_dis_share, opt):

    #reconstruct loss
    reconstruction_loss_hc = MSELoss()(data1.x, out_hc)
    reconstruction_loss_dis_share = MSELoss()(data2.x, out_dis_hc)
    reconstruction_loss_dis = MSELoss()(data2.x, out_dis)
    reconstruction_loss = reconstruction_loss_hc + reconstruction_loss_dis_share + reconstruction_loss_dis
    # reconstruction_loss = reconstruction_loss_hc + reconstruction_loss_dis
    
    # loss for [1,0] normal distribution, useing mean instead of sum to reduce the scaler effect
    kl_loss_hc = torch.mean(torch.exp(log_var_hc) + mu_hc**2 - 1.0 - log_var_hc)
    kl_loss_dis_share = torch.mean(torch.exp(log_var_dis_hc) + mu_dis_hc**2 - 1.0 - log_var_dis_hc)
    kl_loss_dis = torch.mean(torch.exp(log_var_dis) + mu_dis**2 - 1.0 - log_var_dis)
    kl_loss = kl_loss_hc + kl_loss_dis_share + kl_loss_dis

    fea = torch.cat((y_prob_dis, y_prob_dis_share), 0).type(torch.float)
    label = torch.zeros((fea.shape[0])).type(torch.float) 
    label[:len(z_dis)] = 1.0
    device = torch.device("cuda:0")
    label = label.to(device)

    distengle_loss1 = F.binary_cross_entropy(fea.squeeze(), label)
    del label
    
    # contrastive_loss = similar_loss + distengle_loss1 + distengle_loss2
    contrastive_loss = distengle_loss1


    # regression loss
    predict_dis_loss = torch.sqrt(MSELoss()(y_pred_dis[~data2.pcd[:,3].isnan()].squeeze(), 
                                      torch.log(data2.pcd[~data2.pcd[:,3].isnan(),3]).squeeze()))

    predict_loss = predict_dis_loss

    loss_all = reconstruction_loss + opt.beta*kl_loss + opt.alpha*contrastive_loss  + opt.theta*predict_loss 
    
    return loss_all, reconstruction_loss, opt.beta*kl_loss, opt.alpha*contrastive_loss, predict_loss
    
def contrastive_effect(with_contrast = True, trial = 2):
    '''
    Parameters
    ----------
    task : contrastive_effect: for contrast loss exploration
        DESCRIPTION. The default is 'contrastive_effect'.
    with_contrast : TYPE, bool
        DESCRIPTION. The default is True. when task is not contrastive_effect, we just set it True 
    trial : TYPE, int
        DESCRIPTION. Which trial we want to analyze

    Returns
    -------
    kls : TYPE
        DESCRIPTION.

    '''
    if torch.cuda.is_available():
        setup_seed(trial) 
    kf = KFold(n_splits = opt.fold, shuffle=True)
    i = 0
    kls = []
    si_all = []
    if opt.fold:
        ########all cv
        for index1, index2 in zip(kf.split(hc_target_idx), kf.split(ad_data)):
            i = i + 1
                ############### Define Graph Deep Learning Network ##########################
            if opt.build_net:
                # model = GAE(opt).to(device)
                model = ContrativeNet_infomax(opt).to(device)
                # model = GraphUNet(opt.in_channels, opt.hidden_channels, opt.in_channels,3).to(device)
            if opt.retrain:
                # model = torch.load(opt.model_file)
                checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
                # pretrained_dict = model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
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
                
            # te_idx = sio.loadmat(os.path.join(opt.result_file, '{}.mat'.format(i)))['idx_te_sampling']
            # patient_loader = DataLoader(ad_data.index_select(list(te_idx.squeeze())),batch_size=opt.batchSize,shuffle = True)
            patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = True)
            HC_data_loader = DataLoader(HC_data,batch_size=opt.batchSize,shuffle = True)
            model.eval()
            for data in patient_loader:
                data = data.to(device)
                disoder_net = model.disorder_net
                output_dis, h_dis, y_dis = disoder_net(data.x,
                    data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
                hc_net = model.HC_net
                output_hc, h_hc, y_pred_hc = hc_net(data.x, 
                      data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
                if with_contrast:
                    g_dis = torch.reshape(output_dis, (output_dis.shape[0], 2500))
                    g_hc = torch.reshape(output_hc, (output_hc.shape[0], 2500))
                else:
                    g_dis = torch.reshape(output_dis, (output_dis.shape[0], 700))
                    g_hc = torch.reshape(output_hc, (output_hc.shape[0], 700))

                value_dis = torch.matmul(output_dis, h_dis).detach().cpu().numpy().squeeze()
                value_hc = torch.matmul(output_hc, h_dis).detach().cpu().numpy().squeeze()

                # h_dis = value_dis.detach().cpu().numpy().squeeze()
                # h_hc = value_hc.detach().cpu().numpy()
                h_dis = h_dis.detach().cpu().numpy().squeeze()
                h_hc = h_hc.detach().cpu().numpy().squeeze()
                g_dis = g_dis.detach().cpu().numpy().squeeze()
                g_hc = g_hc.detach().cpu().numpy().squeeze()
                # scaler = StandardScaler()
                # h_hc = list(scaler.fit_transform(h_hc))
                # h_hc_all.extend(h_hc)
            # h_dis = np.array(h_dis_all)
            # h_hc = np.array(h_hc_all)
            kl_distance = kl_div(g_dis, g_hc)
            kls.append(kl_distance)
    
            # h = np.r_[h_dis_, h_hc]
            # # model_reduce = PCA(n_components=2)
            # model_reduce= TSNE(n_components=2,
            #   init='random', perplexity=50)
            # h_reduce = model_reduce.fit_transform(h)
            
            model_reduce= TSNE(n_components=2,
                init='random', perplexity=100)
            # model_reduce = PCA(n_components=2)
            reduce_dis = model_reduce.fit_transform(value_dis)
            # model_reduce = PCA(n_components=2)
            # h_dis_ = model_reduce.fit_transform(h_dis_)
            # model_reduce = PCA(n_components=2)
            reduce_hc = model_reduce.fit_transform(value_hc)
            # h_reduce = np.r_[h_dis, h_dis_, h_hc]
            h_reduce = np.r_[reduce_dis, reduce_hc]

            dis_name = ['Speific global' for i in range(reduce_dis.shape[0])]
            # dis_name2 = ['Speific local' for i in range(h_dis.shape[0])]
            hc_name = ['Shared global' for i in range(reduce_hc.shape[0])]
            # h_name = np.r_[np.array(dis_name), np.array(dis_name2), np.array(hc_name)]
            h_name = np.r_[np.array(dis_name), np.array(hc_name)]
            df = {'x': h_reduce[:,0], 'y': h_reduce[:,1], 'name': h_name}
            df = pd.DataFrame(df)
            # colors = ["#87CEFA", 'dodgerblue', "salmon"]
            colors = ["#87CEFA", 'salmon']
            plt.figure(figsize=(10,10)) 
            ax = plt.gca()  
            sns.set_palette(sns.color_palette(colors))
            sns.scatterplot(data=df, x="x", y="y", hue = 'name')
            ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
            ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色   
            min_ = h_reduce.min()
            max_ = h_reduce.max()
            min_ = min_ - abs(min_)/2
            max_ = max_ + abs(max_)/2
            # loca_x = (65 - 12)//5
            loca_x = (max_ - min_)/3
            loca_y = (max_ - min_)/3
            print(loca_x)
            # loca_y = (30 + 10)//5
            # if loca_x > 0 and loca_y > 0:
            x_major_locator=MultipleLocator(loca_x)
            y_major_locator=MultipleLocator(loca_y)
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(y_major_locator)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    
            plt.axis([min_, max_, min_, max_])
            
            si = silhouette_score(h_reduce, h_name)
            si_all.append(si)
            ax.set_title('Silhouette score {:.3}'.format(si), fontdict=font)
            if with_contrast:
                plt.savefig(r'/home/alex/project/CGCN/ptau_code/result/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/visualization/disentangle_loss/TSNE_dislocalVShcglobal_{}_seed_{}.svg'.format(i, trial),
                            format = 'svg', bbox_inches = 'tight')  
            else:
                plt.savefig(r'/home/alex/project/CGCN/ptau_code/result/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/visualization/no_disentangle_loss/TSNE_dislocalVShcglobal_{}_seed_{}.svg'.format(i, trial),
                            format = 'svg', bbox_inches = 'tight')         

    return kls

def node_embedding(trial = 2):
    '''
    Parameters
    ----------
    trial : TYPE, int
        DESCRIPTION. Which trial we want to analyze

    Returns
    -------
    figures and analystic results for node embedding 

    '''
    if torch.cuda.is_available():
        setup_seed(0) 
    kf = KFold(n_splits = opt.fold, shuffle=True)
    i = 0
    hc_all_node_embedding = []
    dis_all_node_embedding = []
    hc_all_node_weight = []
    dis_all_node_weight = []
    if opt.fold:
        ########all cv
        for index1, index2 in zip(kf.split(hc_target_idx), kf.split(ad_data)):
            i = i + 1
                ############### Define Graph Deep Learning Network ##########################
            if opt.build_net:
                model = ContrativeNet_infomax(opt).to(device)
            if opt.retrain:
                checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
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
                
            patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = True)
            model.eval()
            for data in patient_loader:
                data = data.to(device)
                disoder_net = model.disorder_net
                dis_node_weight = disoder_net.target_predict.weight.detach().cpu().numpy().squeeze()
                output_dis, h_dis, y_dis = disoder_net(data.x,
                    data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
                hc_net = model.HC_net
                hc_node_weight = hc_net.target_predict.weight.detach().cpu().numpy().squeeze()

                output_hc, h_hc, y_pred_hc = hc_net(data.x, 
                      data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)

                h_dis = h_dis.detach().cpu().numpy().squeeze()
                h_hc = h_hc.detach().cpu().numpy().squeeze()
                hc_all_node_embedding.append(h_hc)
                dis_all_node_embedding.append(h_dis)
                hc_all_node_weight.append(hc_node_weight)
                dis_all_node_weight.append(dis_node_weight)
    return hc_all_node_embedding, dis_all_node_embedding, hc_all_node_weight, dis_all_node_weight


def shap_explain(trial = 2):
    '''
    Parameters
    ----------
    trial : TYPE, int
        DESCRIPTION. Which trial we want to analyze

    Returns
    -------
    figures and analystic results for node embedding 

    '''
    if torch.cuda.is_available():
        setup_seed(trial) 
    kf = KFold(n_splits = opt.fold, shuffle=True)
    i = 0
    hc_all_node_embedding = []
    dis_all_node_embedding = []
    hc_all_node_weight = []
    dis_all_node_weight = []
    if opt.fold:
        ########all cv
        for index1, index2 in zip(kf.split(hc_target_idx), kf.split(ad_data)):
            i = i + 1
                ############### Define Graph Deep Learning Network ##########################
            if opt.build_net:
                model = ContrativeNet_infomax(opt).to(device)
            if opt.retrain:
                checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
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

            idx1 = np.r_[np.arange(len(HC_data_aug)), np.random.randint(0, len(HC_data_aug), size=(len(ad_data)-len(HC_data_aug), 1)).squeeze()]
            patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = True)
            hc_loader = DataLoader(HC_data_aug.index_select(list(idx1)),batch_size=opt.batchSize,shuffle = True)
            model.eval()
            for data1, data2 in zip(patient_loader, hc_loader):
                data1 = data1.to(device)
                data2 = data2.to(device)
                explainer = Explainer(
                    model=model,
                    algorithm=CaptumExplainer('ShapleyValueSampling'),
                    explanation_type='phenomenon',
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='regression',
                        task_level='graph',
                    ),
                )
                
                # Generate explanation for the node at index `10`:
                # explanation = explainer(data1.x, data1.edge_index, index=1, target = data1.pcd[:,2], edge_attr1=data1.edge_attr, pcd1=data1.pcd, x2=data2.x,
                #                         edge_index2=data2.edge_index, edge_attr2=data2.edge_attr, pcd2=data2.pcd, eyes=data2.eyes,
                #                         batch=data2.batch)
                output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                    output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
                  data2.x, data2.edge_index, data2.edge_attr, data1.pcd, data2.eyes, data2.batch)
                disoder_net = model.disorder_net
                # dis_node_weight = disoder_net.target_predict.weight.detach().cpu().numpy().squeeze()
                # explainer = shap.DeepExplainer(model, data = [data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
                #                                   data2.x, data2.edge_index, data2.edge_attr, data1.pcd, data2.eyes, data2.batch])
                # shap_values = explainer.shap_values([data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
                #                                   data2.x, data2.edge_index, data2.edge_attr, data1.pcd, data2.eyes, data2.batch])
                # output_dis, h_dis, y_dis = disoder_net(data.x,
                #     data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
                # hc_net = model.HC_net
                # hc_node_weight = hc_net.target_predict.weight.detach().cpu().numpy().squeeze()

                # output_hc, h_hc, y_pred_hc = hc_net(data.x, 
                #       data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)

                # h_dis = h_dis.detach().cpu().numpy().squeeze()
                # h_hc = h_hc.detach().cpu().numpy().squeeze()
                # hc_all_node_embedding.append(h_hc)
                # dis_all_node_embedding.append(h_dis)
                # hc_all_node_weight.append(hc_node_weight)
                # dis_all_node_weight.append(dis_node_weight)
    return hc_all_node_embedding, dis_all_node_embedding, hc_all_node_weight, dis_all_node_weight

def permute_node_explain(trial = 2):
    '''
    Parameters
    ----------
    trial : TYPE, int
        DESCRIPTION. Which trial we want to analyze

    Returns
    -------
    figures and analystic results for node embedding 

    '''
    if torch.cuda.is_available():
        setup_seed(0) 
    kf = KFold(n_splits = opt.fold, shuffle=True)
    i = 0
    hc_all_node_embedding = []
    dis_all_node_embedding = []
    hc_all_node_weight = []
    dis_all_node_weight = []
    loss_all_permute = []

    if opt.fold:
        ########all cv
        for index1, index2 in zip(kf.split(hc_target_idx), kf.split(ad_data)):
            print(i)
            i = i + 1
                ############### Define Graph Deep Learning Network ##########################
            if opt.build_net:
                model = ContrativeNet_infomax(opt).to(device)
            if opt.retrain:
                checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
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

            idx1 = np.r_[np.arange(len(HC_data_aug)), np.random.randint(0, len(HC_data_aug), size=(len(ad_data)-len(HC_data_aug), 1)).squeeze()]
            patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = True)
            hc_loader = DataLoader(HC_data_aug.index_select(list(idx1)),batch_size=opt.batchSize,shuffle = True)
            model.eval()
            with torch.no_grad():
    
                for data1, data2 in zip(hc_loader, patient_loader):
                    data1 = data1.to(device)
                    data2 = data2.to(device)
                    output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                        output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
                      data2.x, data2.edge_index, data2.edge_attr, data1.pcd, data2.eyes, data2.batch)
    
                    pos_loss = -torch.log(logits_dis).mean()
                    neg_loss_dis_share = -torch.log(1 -logits_dis_share).mean()
                    neg_loss_hc = -torch.log(1 -logits_hc).mean()
    
                    contrastive_loss = pos_loss + neg_loss_dis_share + neg_loss_hc
                    predict_dis_loss = MSELoss()(y_pred_dis.squeeze(), torch.log(data2.pcd[:,2]).squeeze())
                    predict_loss = opt.theta1 * predict_dis_loss 
                    loss_gt = (predict_loss + opt.alpha*contrastive_loss).squeeze().item() 
                    # y_true_dis = data2.pcd[~data2.pcd[:,2].isnan(),2].detach().cpu().tolist()
                    # y_pred_dis = torch.exp(y_pred_dis).detach().cpu().tolist()
                    # r2 = r2_score(y_true_dis, y_pred_dis)
                    #########################permute
                loss_multi_permute = []
                for data1_, data2_ in zip(hc_loader, patient_loader):
                    data1 = data1_.to(device)
                    data2 = data2_.to(device)
                    for k in range(3):
                        print(k)
                        loss_onerun = []
                        for j in range(100):
                            edge2 = to_dense_adj(data2.edge_index, data2.batch)
                            # idx_permute = np.random.permutation(data2.x.shape[1])
                            idx_permute = np.random.permutation(data2.pcd.shape[0])
                            # edge2[:,j,:] = edge2[idx_permute,j,:]
                            # edge2[:,:,j] = edge2[idx_permute,:,j]
                            # edge2[:,j,:] = edge2[:,j,idx_permute]
                            # edge2[:,:,j] = edge2[:,idx_permute,j]
                            edge2 = dense_to_sparse(edge2)
                            # x, mask = to_dense_batch(data2.x, data2.batch)
                            # x[:,j,:] = x[idx_permute,j,:]
                            # x[:,:,j] = x[idx_permute,:,j]
                            # x = x[mask]
                            idx_permute = np.random.permutation(data2.x.shape[0])
                            x = copy.deepcopy(data2.x)
                            x[:,j] = data2.x[idx_permute,j]
                            # x[:,j,:] = x[:,j,idx_permute]
                            # x[:,:,j] = x[:,idx_permute,j]
                            # edge2[:,j,:] = 0
                            # edge2[:,:,j] = 0
                            # edge2 = dense_to_sparse(edge2)
                            # x, mask = to_dense_batch(data2.x, data2.batch)
                            # x[:,j,:] = 0
                            # x[:,:,j] = 0
                            output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                                output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
                              x, edge2[0], data2.edge_attr, data2.pcd, data2.eyes, data2.batch)
                            pos_loss = -torch.log(logits_dis).mean()
                            neg_loss_dis_share = -torch.log(1 -logits_dis_share).mean()
                            neg_loss_hc = -torch.log(1 -logits_hc).mean()
            
                            contrastive_loss = pos_loss + neg_loss_dis_share + neg_loss_hc
                            predict_dis_loss = MSELoss()(y_pred_dis.squeeze(), torch.log(data2.pcd[:,2]).squeeze())
                            predict_loss = opt.theta1 * predict_dis_loss 
                            loss_permute = (predict_loss + opt.alpha*contrastive_loss).squeeze().item() 
                            # loss_permute = MSELoss()(torch.exp(y_pred_dis[~data2.pcd[:,2].isnan()]).squeeze(), 
                            #     data2.pcd[~data2.pcd[:,2].isnan(),2]).squeeze().item() 
                            loss_onerun.append(loss_permute)
                        loss_multi_permute.append(loss_onerun)
                loss_all_permute.append(loss_multi_permute)
    return np.array(loss_all_permute), loss_gt

def permute_edge_explain_all_loss(trial = 2):
    '''
    Parameters
    ----------
    trial : TYPE, int
        DESCRIPTION. Which trial we want to analyze

    Returns
    -------
    figures and analystic results for node embedding 

    '''
    if torch.cuda.is_available():
        setup_seed(0) 
    kf = KFold(n_splits = opt.fold, shuffle=True)
    i = 0
    loss_permute = []
    if opt.fold:
        ########all cv
        for index1, index2 in zip(kf.split(hc_target_idx), kf.split(ad_data)):
            print(i)
            i = i + 1
                ############### Define Graph Deep Learning Network ##########################
            if opt.build_net:
                model = ContrativeNet_infomax(opt).to(device)
            if opt.retrain:
                checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cpu"))
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

            patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = True)
            idx1 = np.r_[np.arange(len(HC_data_aug)), np.random.randint(0, len(HC_data_aug), size=(len(ad_data)-len(HC_data_aug), 1)).squeeze()]
            hc_loader = DataLoader(HC_data_aug.index_select(list(idx1)),batch_size=opt.batchSize,shuffle = True)
            model.eval()
            with torch.no_grad():
    
                for data1, data2 in zip(hc_loader, patient_loader):
    
                    data1 = data1.to(device)
                    data2 = data2.to(device)
                    output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                        output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
                      data2.x, data2.edge_index, data2.edge_attr, data2.pcd, data2.eyes, data2.batch)
                    pos_loss = -torch.log(logits_dis).mean()
                    neg_loss_dis_share = -torch.log(1 -logits_dis_share).mean()
                    neg_loss_hc = -torch.log(1 -logits_hc).mean()
    
                    contrastive_loss = pos_loss + neg_loss_dis_share + neg_loss_hc
                    predict_dis_loss = MSELoss()(y_pred_dis.squeeze(), torch.log(data2.pcd[:,2]).squeeze())
                    predict_loss = opt.theta1 * predict_dis_loss 
                    loss_gt = predict_loss + opt.alpha*contrastive_loss
                    # local_dis_specific_gt = output_dis.reshape(261, -1).detach().cpu().numpy().squeeze()
                    # local_dis_shared_gt = output_dis_share.reshape(261, -1).detach().cpu().numpy().squeeze()
    
                    #########################permute
                for data1_, data2_ in zip(hc_loader, patient_loader):
                    data1 = data1_.to(device)
                    data2 = data2_.to(device)
                    # distance_dis_spec_feas = []
                    loss_onerun = []
                    idx = np.triu_indices_from(np.zeros((100,100)), 1)  
                    for j in range(len(idx[0])):
                        print(j)
                        ##########permute dis
                        edge2 = to_dense_adj(data2.edge_index, data2.batch)
                        idx_permute = np.random.permutation(data2.pcd.shape[0])
                        edge2[:,idx[0][j],idx[1][j]] = edge2[idx_permute,idx[0][j],idx[1][j]]
                        edge2[:,idx[1][j],idx[0][j]] = edge2[idx_permute,idx[1][j],idx[0][j]]
                        edge2 = dense_to_sparse(edge2)
                        x2, mask = to_dense_batch(data2.x, data2.batch)
                        x2[:,idx[0][j],idx[1][j]] = x2[idx_permute,idx[0][j],idx[1][j]]
                        x2[:,idx[0][j],idx[1][j]] = x2[idx_permute,idx[0][j],idx[1][j]]
                        x2 = x2[mask]
                        output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                            output_hc, h_hc, y_pred_hc, logits_hc = model(data1.x, data1.edge_index, data1.edge_attr, data1.pcd, 
                          x2, edge2[0], data2.edge_attr, data2.pcd, data2.eyes, data2.batch)
                        pos_loss = -torch.log(logits_dis).mean()
                        neg_loss_dis_share = -torch.log(1 -logits_dis_share).mean()
                        neg_loss_hc = -torch.log(1 -logits_hc).mean()
    
                        contrastive_loss = pos_loss + neg_loss_dis_share + neg_loss_hc
                        predict_dis_loss = MSELoss()(y_pred_dis.squeeze(), torch.log(data2.pcd[:,2]).squeeze())
                        predict_loss = opt.theta1 * predict_dis_loss 
                        loss_dis_permute = predict_loss + opt.alpha*contrastive_loss
    
                        ##########permute HC
                        edge1 = to_dense_adj(data1.edge_index, data1.batch)
                        edge1[:,idx[0][j],idx[1][j]] = edge1[idx_permute,idx[0][j],idx[1][j]]
                        edge1[:,idx[1][j],idx[0][j]] = edge1[idx_permute,idx[1][j],idx[0][j]]
                        edge1 = dense_to_sparse(edge1)
                        x1, mask = to_dense_batch(data1.x, data1.batch)
                        x1[:,idx[0][j],idx[1][j]] = x1[idx_permute,idx[0][j],idx[1][j]]
                        x1[:,idx[0][j],idx[1][j]] = x1[idx_permute,idx[0][j],idx[1][j]]
                        x1 = x1[mask]
                        output_dis, h_dis, y_pred_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
                            output_hc, h_hc, y_pred_hc, logits_hc = model(x1, edge1[0], data1.edge_attr, data1.pcd, 
                          data2.x, data2.edge_index, data2.edge_attr, data2.pcd, data2.eyes, data2.batch)
                        pos_loss = -torch.log(logits_dis).mean()
                        neg_loss_dis_share = -torch.log(1 -logits_dis_share).mean()
                        neg_loss_hc = -torch.log(1 -logits_hc).mean()
                        
                        contrastive_loss = pos_loss + neg_loss_dis_share + neg_loss_hc
                        predict_dis_loss = MSELoss()(y_pred_dis.squeeze(), torch.log(data2.pcd[:,2]).squeeze())
                        predict_loss = opt.theta1 * predict_dis_loss 
                        loss_hc_permute = predict_loss + opt.alpha*contrastive_loss
                        optimizer.zero_grad()
                        # local_dis_specific_permute = output_dis.reshape(261, -1).detach().cpu().numpy().squeeze()
                        # local_dis_shared_permute = output_dis_share.reshape(261, -1).detach().cpu().numpy().squeeze()
                        # distance_dis_spec = cdist(local_dis_specific_gt, local_dis_specific_permute)
                        # distance_dis_spec = np.diagonal(distance_dis_spec).mean()
                        # distance_dis_share = cdist(local_dis_shared_gt, local_dis_shared_permute)
                        # distance_dis_share = np.diagonal(distance_dis_share).mean()
                        # distance_dis_spec_feas.append(distance_dis_spec)
                        # distance_dis_share_feas.append(distance_dis_share)
                        loss_onerun.append([(loss_dis_permute-loss_gt).cpu().numpy().squeeze(), (loss_hc_permute-loss_gt).cpu().numpy().squeeze()])
                    loss_permute.append(loss_onerun)
    return np.array(loss_permute)

if __name__ == '__main__':
    
    ###################################################
    ###################################################ras all A+
    ###################################################
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    with_contrast = True
    hc_all_node_embeddings, dis_all_node_embeddings, hc_all_node_weights, dis_all_node_weights = [], [], [], []
    trials = range(10)
    v_p_all_hc = []
    v_p_all_dis = []
    v_p_all_hc_rsa = []
    v_p_all_dis_rsa = []
    y_dis_mean_all = []
    y_hc_mean_all = []
    dis_latent_all = []
    hc_latent_all = []
    for trial in trials:
        parser = argparse.ArgumentParser()
        if with_contrast:
            parser.add_argument('--model_file', type=str, default=\
                                r'/home/alex/project/CGCN/A+/ab/model/seed2/0.2/{}'.format(trial), 
                                help='model save path')
            parser.add_argument('--result_file', type=str, default=\
                                r'/home/alex/project/CGCN/A+/ab/result/seed2/0.2/{}'.format(trial), 
                                help='result save path')
        else:
            parser.add_argument('--model_file', type=str, default=\
                                r'/home/alex/project/CGCN/ptau_code/model/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
                                help='model save path')
            parser.add_argument('--result_file', type=str, default=\
                                r'/home/alex/project/CGCN/ptau_code/result/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
                                help='result save path')
        parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs of training')
        parser.add_argument('--batchSize', type=int, default= 700, help='size of the batches')
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
        parser.add_argument('--act', type=str, default='relu', help='relu || leaky_relu || prelu || tanh')
        parser.add_argument('--sum_res', type=bool, default=True)
        parser.add_argument('--save_model', action='store_true')
        parser.add_argument('--normalization', action='store_true') 
        parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
        parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
        parser.add_argument('--dataroot', type=str,
                            default=r'/home/alex/project/CGCN/A+/ptau/data/',
                            help='root directory of the dataset')
        parser.add_argument('--retrain', default=True, type=bool, help='whether train from used model')     
        parser.add_argument('--epsilon', default=0.1, type=float, help='stochastic epsilon for gcn')
        parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
        parser.add_argument('--demean', type=bool, default=True)
        parser.add_argument('--drop', default=0.3, type=float, help='drop ratio')
        parser.add_argument('--task', default='regression_hc_visual', type=str, help='classfication / regression/regression_hc_visual/classification_hc_visual')
        parser.add_argument('--augmentation', default=10, type=int, help='times of augmentation')
        parser.add_argument('--cluster', default=7, type=int, help='cluster number')

        parser.set_defaults(normalization=True)
        opt = parser.parse_args()
        name = 'Biopoint'
    
        dataset = BiopointDataset(opt, name)
        HC_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0]
        ad_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0]
        hc_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),2][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0].numpy()
        hc_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]==0,2].numpy())
        hc_target_idx = np.where(~np.isnan(hc_target))[0]
        ad_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),3][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
        ad_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]!=0,2].numpy())
        ad_target_idx = np.where(~np.isnan(ad_target))[0]
        HC_data_aug = dataset[dataset.data.pcd[:,-1]==0]
        ad_data_aug = dataset[dataset.data.pcd[:,-1]!=0]
        ################################## node embedding
        prevent_pcd = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
        prevent_pcd_used = prevent_pcd[['CONP_CandID', 'Candidate_Age', 'Gender', 'AD8_total_score', 'Systolic_blood_pressure',
                                        'Diastolic_blood_pressure', 'tau', 'ptau', 'Amyloid_beta_1_42', 'G_CSF', 'IL_15', 'IL_8', 'VEGF',
                                        'APOE', 'BchE_K_variant', 'BDNF', 'HMGCR_Intron_M', 'TLR4_rs_4986790', 'PPP2r1A_rs_10406151',
                                        'CDK5RAP2_rs10984186', 'immediate_memory_index_score', 'visuospatial_constructional_index_score', 
                                        'language_index_score', 'attention_index_score', 'delayed_memory_index_score', 'total_scale_index_score',]]
        prevent_demo = pd.read_csv('/home/alex/project/CGCN/dataset/AD/PREVENT-AD/PhenotypicData/Demographics_Registered_PREVENTAD.csv')[[
            'CONP_CandID', 'Sex', 'Ethnicity', 'Education_years', 'father_dx_ad_dementia', 'mother_dx_ad_dementia', 'sibling_dx_ad_dementia', 'other_family_members_AD']]
        prevent_demo = prevent_demo.rename(columns={'Sex':'Gender'})
        prevent_demo['Ethnicity'][prevent_demo['Ethnicity'] == 'caucasian'] = 'White'
        prevent_demo['Ethnicity'][prevent_demo['Ethnicity'] == 'other'] = 'Unknown'

        prevent_pcd_used = pd.merge(prevent_pcd_used, prevent_demo, how='outer', on=['CONP_CandID', 'Gender'])
        prevent_pcd_used['Candidate_Age'] = prevent_pcd_used['Candidate_Age'] / 12
        adni_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
        adni_pcd_used = adni_pcd[['subjectID', 'AGE', 'gender', 'MOCA', 'ADAS11', 'ADAS13', 'PTEDUCAT', 'PTRACCAT', 'TAU', 'PTAU', 
                                  'ABETA', 'CDRSB', 'MMSE', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
                                  'EcogPtTotal', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting']]
        adni_pcd_used['Systolic_blood_pressure'] = np.nan
        adni_pcd_used['Diastolic_blood_pressure'] = np.nan
        adni_dx = adni_pcd['DX']
        adni_pcd_used['subjectID'] = adni_pcd_used['subjectID'].str[6:].astype(int)
        adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Male'] = 'M'
        adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Female'] = 'F'
        adni_bp = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/AV45VITALS_20Aug2023.csv')[['RID', 'VISCODE2', 'PRESYSTBP', 'PREDIABP']]
        adni_bp_bl = adni_bp[adni_bp['VISCODE2'] == 'bl']
        [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_bp_bl['RID'], return_indices=True)
        adni_pcd_used.iloc[IA,-2] = adni_bp_bl.iloc[IB]['PRESYSTBP']
        adni_pcd_used.iloc[IA,-1] = adni_bp_bl.iloc[IB]['PREDIABP'] #no IL factors accessible
        
        adni_apoe = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/APOERES_20Aug2023.csv')[['RID', 'APGEN1', 'APGEN2']]
        adni_pcd_used['APOE'] = np.nan
        [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_apoe['RID'], return_indices=True)
        adni_pcd_used.iloc[IA,-1] = adni_apoe.iloc[IB]['APGEN1'].astype('str') + ' ' + adni_apoe.iloc[IB]['APGEN2'].astype('str')
        prevent_pcd_used = prevent_pcd_used.rename(columns={'CONP_CandID': 'subjectID', 'Candidate_Age': 'AGE', 'Gender': 'gender', 'tau': 'TAU', 'ptau': 'PTAU', 
                                        'Amyloid_beta_1_42': 'ABETA', 'Education_years': 'PTEDUCAT', 'Ethnicity': 'PTRACCAT'})
        prevent_pcd_used['gender'][prevent_pcd_used['gender'] == 'Male'] = 'M'
        prevent_pcd_used['gender'][prevent_pcd_used['gender'] == 'Female'] = 'F'
        prevent_dx = np.array(['CN' for i in range(len(prevent_pcd_used))])
        pcd_all = pd.merge(adni_pcd_used, prevent_pcd_used, how='outer', on=['subjectID', 'AGE', 'gender', 'TAU', 'PTAU', 'ABETA', 'PTEDUCAT', 'APOE', 
                                                                              'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'PTRACCAT'])
        pcd_mri = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),:][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
        dx_all = np.r_[adni_dx, prevent_dx]

        sub, IA, IB = np.intersect1d(pcd_mri[:,0], pcd_all['subjectID'], return_indices=True)
        _, IAA, IBB = np.unique(IA, return_index=True, return_inverse=True)
        pcd_all_used = pcd_all.iloc[IB].iloc[IAA].reset_index(drop=True)
        keys_raw = pcd_all_used.columns.values
        dx_used = dx_all[IB][IAA]
        
        scaler = StandardScaler()
        if torch.cuda.is_available():
            setup_seed(0) 
        kf = KFold(n_splits = opt.fold, shuffle=True)
        i = 0
        hc_all_node_embedding = []
        dis_all_node_embedding = []
        hc_all_node_weight = []
        dis_all_node_weight = []
        y_dis_all = []
        y_hc_all = []
        v_p_hc = {}
        v_p_dis = {}
        v_p_hc_rsa = {}
        v_p_dis_rsa = {}
        ########all cv
        for index in kf.split(ad_data):
            i = i + 1
                ############### Define Graph Deep Learning Network ##########################
            if opt.build_net:
                model = ContrativeNet_infomax(opt).to(device)
            if opt.retrain:
                checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
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
                # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
                scheduler.load_state_dict(checkpoint['scheduler'])  # 加载优化器参数
                
            patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = False)
            model.eval()
            for data in patient_loader:
                data = data.to(device)
                disoder_net = model.disorder_net
                dis_node_weight = disoder_net.target_predict.weight.detach().cpu().numpy().squeeze()
                output_dis, h_dis, y_dis = disoder_net(data.x,
                    data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
                hc_net = model.HC_net    
                output_hc, h_hc, y_pred_hc = hc_net(data.x,
                      data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
                y_dis = y_dis.detach().cpu().numpy().squeeze()
                y_pred_hc = y_pred_hc.detach().cpu().numpy().squeeze()
                h_dis = h_dis.detach().cpu().numpy().squeeze()
                h_hc = h_hc.detach().cpu().numpy().squeeze()
            y_dis_all.append(y_dis)
            y_hc_all.append(y_pred_hc)
            dis_all_node_embedding.append(h_dis)
            hc_all_node_embedding.append(h_hc)
        y_dis_all = np.array(y_dis_all)
        y_hc_all = np.array(y_hc_all)
        dis_all_node_embedding = np.array(dis_all_node_embedding)
        hc_all_node_embedding = np.array(hc_all_node_embedding)
        y_dis_mean = y_dis_all.mean(0)
        y_hc_mean = y_hc_all.mean(0)
        if opt.task == 'classification':
            y_dis_mean = y_dis_mean[:,0]
            y_hc_mean = y_hc_mean[:,0]
        y_dis_mean_all.append(y_dis_all)
        y_hc_mean_all.append(y_hc_all)
        dis_latent_all.append(dis_all_node_embedding)
        hc_latent_all.append(hc_all_node_embedding)
        hc_latent_mean = hc_all_node_embedding.mean(0)
        dis_latent_mean = dis_all_node_embedding.mean(0)
        # hc_latent_rep = cdist(hc_latent_mean, hc_latent_mean, 'correlation')
        for j in range(pcd_all_used.shape[-1]):
            # j = 7
            mask = ~pcd_all_used.iloc[:,j].isnull()
            feature = pcd_all_used.iloc[:,j][mask]
            y_dis = y_dis_mean[mask]
            y_hc = y_hc_mean[mask]
            hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
            dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
            hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
            dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
            if j == 0 or (mask).sum()<15:
                continue
            else:
                if keys_raw[j] == 'CDRSB':
                    feature = pcd_all_used.iloc[:,j]
                    feature[~mask] = 0
                    out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
                    out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
                    method = 0
                    size = len(y_dis_mean)
                    feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                    feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
                    hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
                    dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
                    hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
                    dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
                    
                elif keys_raw[j] == 'PTRACCAT':
                    mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
                    feature = feature[mask2]
                    hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
                    dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
                    hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
                    dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
                    feature[feature == 'White'] = '0'
                    feature[feature == 'Black'] = '1'
                    feature[feature == 'Asian'] = '2'
                    feature[feature == 'More than one'] = '3'
                    feature = feature.astype(float)
                    out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
                    out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
                    method = 0
                    size = len(y_dis)
                    feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                    feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
                elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
                    or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
                    out_dis = spearmanr(feature, y_dis)
                    out_hc = spearmanr(feature, y_hc)
                    method = 1
                    size = len(y_dis)
                    feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                    feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

                elif keys_raw[j] == 'APOE':
                    count2 = feature.str.count('2')
                    mask2 = count2 == 0
                    feature = feature[mask2]
                    hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
                    dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
                    hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
                    dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
                    count3 = feature.str.count('3')
                    count4 = feature.str.count('4')
                    # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
                    apoe_risk = count3 * 0 + count4 * 1
                    apoe_risk = count4
                    out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
                    out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
                    method = 0
                    size = len(y_dis)
                    feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
                    feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

                elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
                    or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
                    or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
                    or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
                    values = pd.unique(feature)
                    feature_new = np.zeros((len(feature)))
                    if isinstance(values[0], str):
                        for k in range(len(values)):
                            feature_new[feature == values[k]] = k
                    else:
                        feature_new = feature
                        
                    if len(values) == 2:
                        out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
                        out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
                    elif len(values) == 3:
                        out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
                                            y_dis[feature==values[2]])
                        out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
                                          y_hc[feature==values[2]])  
                    method = 0
                    size = len(y_dis)
                    feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
                    feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

                else:
                    out_dis = pearsonr(feature, y_dis)
                    out_hc = pearsonr(feature, y_hc)
                    method = 2
                    size = len(y_dis)
                    feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                    feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

            out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
            out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
            np.fill_diagonal(hc_latent_rep_cos, 0)
            np.fill_diagonal(dis_latent_rep_cos, 0)
            out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
            out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)

            v_p_hc[keys_raw[j]] = [out_hc[0], out_hc[1], method, size]
            v_p_dis[keys_raw[j]] = [out_dis[0], out_dis[1], method, size]     
            v_p_hc_rsa[keys_raw[j]] = [out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]]
            v_p_dis_rsa[keys_raw[j]] = [out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]]     
        v_p_all_hc.append(v_p_hc)
        v_p_all_dis.append(v_p_dis) 
        v_p_all_hc_rsa.append(v_p_hc_rsa)
        v_p_all_dis_rsa.append(v_p_dis_rsa)
    v_p_hc_array = np.zeros((10, 47, 8))
    v_p_dis_array = np.zeros((10, 47, 8))
    keys = list(v_p_all_hc[1].keys())
    for j in range(len(v_p_all_hc)):
        for k in range(len(v_p_all_hc[1])):
            v_p_hc_array[j, k, :4] = v_p_all_hc[j][keys[k]] 
            v_p_dis_array[j, k, :4] = v_p_all_dis[j][keys[k]] 
            v_p_hc_array[j, k, 4:] = v_p_all_hc_rsa[j][keys[k]] 
            v_p_dis_array[j, k, 4:] = v_p_all_dis_rsa[j][keys[k]] 
            
    y_dis_mean_all = np.array(y_dis_mean_all)
    y_hc_mean_all = np.array(y_hc_mean_all)
    y_dis_mean_f = y_dis_mean_all.mean(0)
    y_hc_mean_f = y_hc_mean_all.mean(0)
    dis_latent_all = np.array(dis_latent_all)
    hc_latent_all = np.array(hc_latent_all)
    v_p_dis_f = []
    v_p_hc_f = []
    v_p_dis_f_rsa = []
    v_p_hc_f_rsa = []
    ########## association of averaged model outcomes
    for j in range(pcd_all_used.shape[-1]):
        mask = ~pcd_all_used.iloc[:,j].isnull()
        feature = pcd_all_used.iloc[:,j][mask]
        y_dis = y_dis_mean[mask]
        y_hc = y_hc_mean[mask]
        hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
        dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
        hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
        dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
        
        if j == 0 or (mask).sum()<15:
            continue
        else:
            if keys_raw[j] == 'CDRSB':
                feature = pcd_all_used.iloc[:,j]
                feature[~mask] = 0
                out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
                out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
                method = 0
                size = len(y_dis_mean)
                feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
                dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
                hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
                dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
                feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

            elif keys_raw[j] == 'PTRACCAT':
                mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
                feature = feature[mask2]
                hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
                dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
                hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
                dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
                feature[feature == 'White'] = '0'
                feature[feature == 'Black'] = '1'
                feature[feature == 'Asian'] = '2'
                feature[feature == 'More than one'] = '3'
                feature = feature.astype(float)
                out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
                out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
                method = 0
                size = len(y_dis)
                feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
                    
            elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
                or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
                out_dis = spearmanr(feature, y_dis)
                out_hc = spearmanr(feature, y_hc)
                method = 1
                size = len(y_dis)
                feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

            elif keys_raw[j] == 'APOE':
                count2 = feature.str.count('2')
                mask2 = count2 == 0
                feature = feature[mask2]
                hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
                dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
                hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
                dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
                count3 = feature.str.count('3')
                count4 = feature.str.count('4')
                # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
                apoe_risk = count3 * 0 + count4 * 1
                apoe_risk = count4
                out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
                out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
                method = 0
                size = len(y_dis)
                feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
                feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

                # count2 = feature.str.count('2')
                # count3 = feature.str.count('3')
                # count4 = feature.str.count('4')
                # # apoe_risk = 0 * count2 + count3 * 1 + count4 *2
                # apoe_risk = count4
                # out_dis = f_oneway(y_dis[apoe_risk>0], y_dis[apoe_risk<=0])
                # out_hc = f_oneway(y_hc[apoe_risk>0], y_hc[apoe_risk<=0])
                # method = 0
                # size = len(y_dis)
                # feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
                # feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

            elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
                or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
                or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
                or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
                values = pd.unique(feature)
                feature_new = np.zeros((len(feature)))
                if isinstance(values[0], str):
                    for k in range(len(values)):
                        feature_new[feature == values[k]] = k
                else:
                    feature_new = feature
                if len(values) == 2:
                    out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
                    out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
                elif len(values) == 3:
                    out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
                                        y_dis[feature==values[2]])
                    out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
                                      y_hc[feature==values[2]])  
                method = 0
                size = len(y_dis)
                feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
                feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

            else:
                out_dis = pearsonr(feature, y_dis)
                out_hc = pearsonr(feature, y_hc)
                method = 2
                size = len(y_dis)
                feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
                feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

        # weight_half_dis = keep_triangle_half(dis_latent_rep.shape[0]*(dis_latent_rep.shape[0]-1)//2, 1, np.expand_dims(dis_latent_rep, 0)).squeeze()
        # weight_half_fea = keep_triangle_half(feature_rep.shape[0]*(feature_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
        # weight_half = keep_triangle_half(hc_latent_rep.shape[0]*(hc_latent_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
        np.fill_diagonal(hc_latent_rep_cos, 0)
        np.fill_diagonal(dis_latent_rep_cos, 0)
        out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
        out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
        out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
        out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)
        v_p_hc_f.append([out_hc[0], out_hc[1], method, size])
        v_p_dis_f.append([out_dis[0], out_dis[1], method, size])
        v_p_hc_f_rsa.append([out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]])
        v_p_dis_f_rsa.append([out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]])

    v_p_hc_f = np.array(v_p_hc_f)
    v_p_dis_f = np.array(v_p_dis_f)
    v_p_hc_f_rsa = np.array(v_p_hc_f_rsa)
    v_p_dis_f_rsa = np.array(v_p_dis_f_rsa)
    v_p_hc_f_ = np.delete(v_p_hc_f, [7,8,9],0) #for prediction analysis
    v_p_dis_f_ = np.delete(v_p_dis_f, [7,8,9],0)
    v_p_hc_f_rsa_ = np.delete(v_p_hc_f_rsa, [7,8,9],0)
    v_p_dis_f_rsa_ = np.delete(v_p_dis_f_rsa, [7,8,9],0)
    keys_fdr = np.delete(np.array(keys), [7,8,9],0)
    ########## association of averaged model outcomes
    r_p_pcd_all = []
    for j in range(pcd_all_used.shape[-1]):
        mask = ~pcd_all_used.iloc[:,j].isnull()
        r_p_pcd = []
        for k in range(1):
            # mask2 = ~pcd_all_used.iloc[:,k+1].isnull()
            feature = pcd_all_used.iloc[:,j][mask]
            feature2 = data.pcd[:,0].detach().cpu().numpy().squeeze()[mask]
            
            if keys_raw[j] == 'APOE':
                count2 = feature.str.count('2')
                count3 = feature.str.count('3')
                count4 = feature.str.count('4')
                feature = -1 * count2 + count3 * 0 + count4
            elif keys_raw[j] == 'CDRSB':
                feature[feature !=0] = 1
                print(feature)
                print(j)
            else:
                if isinstance(feature.iloc[0],str):
                    values = pd.unique(feature)
                    feature_new = np.zeros((len(feature)))
                    if isinstance(values[0], str):
                        for l in range(len(values)):
                            feature_new[feature == values[l]] = l
                    feature = feature_new
            size = len(feature)
            out = stats.pointbiserialr(feature, feature2)
            r_p_pcd.append([out[0], out[1], size])
        r_p_pcd_all.append(r_p_pcd)

    r_p_pcd_all = np.array(r_p_pcd_all)
    # sio.savemat('/home/alex/project/CGCN/A+/final_figure/ab/statistics_withPCDassociation_withrace_A+all_relu.mat', {'dis_mean_final_stats': v_p_dis_f, 'hc_mean_final_stats': v_p_hc_f,
    #           'keys_all': keys, 'dis_all_final_stats': v_p_dis_array, 'hc_all_final_stats': v_p_hc_array, 'y_dis_ten_trials': y_dis_mean_all, 
    #           'y_hc_ten_trials': y_hc_mean_all, 'dis_latent_all': dis_latent_all, 'hc_latent_all': hc_latent_all, 'dis_latent_features': dis_latent_all,
    #           'hc_latent_features': hc_latent_all, 'hc_mean_final_stats_rsa': v_p_hc_f_rsa, 'dis_mean_final_stats_rsa': v_p_dis_f_rsa, 
    #           'hc_notarget_stats': v_p_hc_f_, 'dis_notarget_stats': v_p_dis_f_, 'hc_notarget_stats_rsa': v_p_hc_f_rsa_, 'dis_notarget_stats_rsa':
    #               v_p_dis_f_rsa_, 'keys_notarget': keys_fdr, 'r_p_pcd': r_p_pcd_all})
    # pcd_all_used['DX'] = dx_used
    # pcd_all_used.to_csv('/home/alex/project/CGCN/A+/final_figure/ab/pcd_feature_withrace_A+all_relu.csv')

    # ###################################################
    # ###################################RSA A+ NC
    # ##############################################
    # device = torch.device("cuda:0")
    # # device = torch.device("cpu")
    # with_contrast = True
    # hc_all_node_embeddings, dis_all_node_embeddings, hc_all_node_weights, dis_all_node_weights = [], [], [], []
    # trials = [1,11,2,14,0,8,7,6,19,5]
    # # trials = range(10)
    # v_p_all_hc = []
    # v_p_all_dis = []
    # v_p_all_hc_rsa = []
    # v_p_all_dis_rsa = []
    # y_dis_mean_all = []
    # y_hc_mean_all = []
    # dis_latent_all = []
    # hc_latent_all = []
    # for trial in trials:
    #     parser = argparse.ArgumentParser()
    #     if with_contrast:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/model/seed2/0.2/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/result/seed2/0.2/{}'.format(trial), 
    #                             help='result save path')
    #     else:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/model/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/result/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='result save path')
    #     parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs of training')
    #     parser.add_argument('--batchSize', type=int, default= 700, help='size of the batches')
    #     parser.add_argument('--fold', type=int, default=5, help='training which fold')
    #     parser.add_argument('--lr', type = float, default=0.1, help='learning rate')
    #     parser.add_argument('--stepsize', type=int, default=250, help='scheduler step size')
    #     # parser.add_argument('--stepsize', type=int, default=22, help='scheduler step size')
    #     parser.add_argument('--weightdecay', type=float, default=0.01, help='regularization')
    #     # parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
    #     parser.add_argument('--gamma', type=float, default=0.4, help='scheduler shrinking rate')
    #     parser.add_argument('--alpha', type=float, default=1, help='loss control to disentangle HC and disorder')
    #     parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
    #     parser.add_argument('--beta', type=float, default=1, help='loss control to force gaussian distribution')
    #     parser.add_argument('--theta1', type=float, default=1, help='loss control to prediction task for dis')
    #     parser.add_argument('--theta2', type=float, default=0.2, help='loss control to prediction task for HC')
    #     parser.add_argument('--build_net', default=True, type=bool, help='model name')
    #     parser.add_argument('--in_channels', type=int, default=100)
    #     parser.add_argument('--hidden_channels', type=int, default=50)
    #     parser.add_argument('--depth', type=int, default=1)
    #     parser.add_argument('--conv', type=str, default='edgeconv', help='edgeconv || gat || gcn || gen')
    #     parser.add_argument('--act', type=str, default='tanh', help='relu || leaky_relu || prelu || tanh')
    #     parser.add_argument('--sum_res', type=bool, default=True)
    #     parser.add_argument('--save_model', action='store_true')
    #     parser.add_argument('--normalization', action='store_true') 
    #     parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
    #     parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    #     parser.add_argument('--dataroot', type=str,
    #                         default=r'/home/alex/project/CGCN/A+/ptau/data/',
    #                         help='root directory of the dataset')
    #     parser.add_argument('--retrain', default=True, type=bool, help='whether train from used model')     
    #     parser.add_argument('--epsilon', default=0.1, type=float, help='stochastic epsilon for gcn')
    #     parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
    #     parser.add_argument('--demean', type=bool, default=True)
    #     parser.add_argument('--drop', default=0.3, type=float, help='drop ratio')
    #     parser.add_argument('--task', default='regression_hc_visual', type=str, help='classfication / regression/regression_hc_visual/classification_hc_visual')
    #     parser.add_argument('--augmentation', default=10, type=int, help='times of augmentation')
    #     parser.add_argument('--cluster', default=7, type=int, help='cluster number')

    #     parser.set_defaults(save_model=True)
    #     parser.set_defaults(normalization=True)
    #     opt = parser.parse_args()
    #     name = 'Biopoint'
    
    #     dataset = BiopointDataset(opt, name)
    #     HC_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0]
    #     ad_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0]
    #     hc_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),2][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0].numpy()
    #     hc_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]==0,2].numpy())
    #     hc_target_idx = np.where(~np.isnan(hc_target))[0]
    #     ad_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),3][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
    #     ad_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]!=0,2].numpy())
    #     ad_target_idx = np.where(~np.isnan(ad_target))[0]
    #     HC_data_aug = dataset[dataset.data.pcd[:,-1]==0]
    #     ad_data_aug = dataset[dataset.data.pcd[:,-1]!=0]
    #     ################################## node embedding
    #     prevent_pcd = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
    #     prevent_pcd_used = prevent_pcd[['CONP_CandID', 'Candidate_Age', 'Gender', 'AD8_total_score', 'Systolic_blood_pressure',
    #                                     'Diastolic_blood_pressure', 'tau', 'ptau', 'Amyloid_beta_1_42', 'G_CSF', 'IL_15', 'IL_8', 'VEGF',
    #                                     'APOE', 'BchE_K_variant', 'BDNF', 'HMGCR_Intron_M', 'TLR4_rs_4986790', 'PPP2r1A_rs_10406151',
    #                                     'CDK5RAP2_rs10984186', 'immediate_memory_index_score', 'visuospatial_constructional_index_score', 
    #                                     'language_index_score', 'attention_index_score', 'delayed_memory_index_score', 'total_scale_index_score',]]
    #     prevent_demo = pd.read_csv('/home/alex/project/CGCN/dataset/AD/PREVENT-AD/PhenotypicData/Demographics_Registered_PREVENTAD.csv')[[
    #         'CONP_CandID', 'Sex', 'Ethnicity', 'Education_years', 'father_dx_ad_dementia', 'mother_dx_ad_dementia', 'sibling_dx_ad_dementia', 'other_family_members_AD']]
    #     prevent_demo = prevent_demo.rename(columns={'Sex':'Gender'})
    #     prevent_demo['Ethnicity'][prevent_demo['Ethnicity'] == 'caucasian'] = 'White'
    #     prevent_demo['Ethnicity'][prevent_demo['Ethnicity'] == 'other'] = 'Unknown'

    #     prevent_pcd_used = pd.merge(prevent_pcd_used, prevent_demo, how='outer', on=['CONP_CandID', 'Gender'])
    #     prevent_pcd_used['Candidate_Age'] = prevent_pcd_used['Candidate_Age'] / 12
    #     adni_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
    #     adni_pcd_used = adni_pcd[['subjectID', 'AGE', 'gender', 'MOCA', 'ADAS11', 'ADAS13', 'PTEDUCAT', 'PTRACCAT', 'TAU', 'PTAU', 
    #                               'ABETA', 'CDRSB', 'MMSE', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
    #                               'EcogPtTotal', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting']]
    #     adni_pcd_used['Systolic_blood_pressure'] = np.nan
    #     adni_pcd_used['Diastolic_blood_pressure'] = np.nan
    #     adni_dx = adni_pcd['DX']
    #     adni_pcd_used['subjectID'] = adni_pcd_used['subjectID'].str[6:].astype(int)
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Male'] = 'M'
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Female'] = 'F'
    #     adni_bp = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/AV45VITALS_20Aug2023.csv')[['RID', 'VISCODE2', 'PRESYSTBP', 'PREDIABP']]
    #     adni_bp_bl = adni_bp[adni_bp['VISCODE2'] == 'bl']
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_bp_bl['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-2] = adni_bp_bl.iloc[IB]['PRESYSTBP']
    #     adni_pcd_used.iloc[IA,-1] = adni_bp_bl.iloc[IB]['PREDIABP'] #no IL factors accessible
        
    #     adni_apoe = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/APOERES_20Aug2023.csv')[['RID', 'APGEN1', 'APGEN2']]
    #     adni_pcd_used['APOE'] = np.nan
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_apoe['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-1] = adni_apoe.iloc[IB]['APGEN1'].astype('str') + ' ' + adni_apoe.iloc[IB]['APGEN2'].astype('str')
    #     prevent_pcd_used = prevent_pcd_used.rename(columns={'CONP_CandID': 'subjectID', 'Candidate_Age': 'AGE', 'Gender': 'gender', 'tau': 'TAU', 'ptau': 'PTAU', 
    #                                     'Amyloid_beta_1_42': 'ABETA', 'Education_years': 'PTEDUCAT', 'Ethnicity': 'PTRACCAT'})
    #     prevent_pcd_used['gender'][prevent_pcd_used['gender'] == 'Male'] = 'M'
    #     prevent_pcd_used['gender'][prevent_pcd_used['gender'] == 'Female'] = 'F'
    #     prevent_dx = np.array(['CN' for i in range(len(prevent_pcd_used))])
    #     pcd_all = pd.merge(adni_pcd_used, prevent_pcd_used, how='outer', on=['subjectID', 'AGE', 'gender', 'TAU', 'PTAU', 'ABETA', 'PTEDUCAT', 'APOE', 
    #                                                                          'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'PTRACCAT'])
    #     dx_all = np.r_[adni_dx, prevent_dx]
    #     pcd_mri = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),:][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
        
    #     sub, IA, IB = np.intersect1d(pcd_mri[:,0], pcd_all['subjectID'], return_indices=True)
    #     _, IAA, IBB = np.unique(IA, return_index=True, return_inverse=True)
    #     pcd_all_used = pcd_all.iloc[IB].iloc[IAA].reset_index(drop=True)
    #     dx_used = dx_all[IB][IAA]
    #     pcd_all_used = pcd_all_used[dx_used == 'CN']
    #     keys_raw = pcd_all_used.columns.values

    #     scaler = StandardScaler()
    #     if torch.cuda.is_available():
    #         setup_seed(0) 
    #     kf = KFold(n_splits = opt.fold, shuffle=True)
    #     i = 0
    #     hc_all_node_embedding = []
    #     dis_all_node_embedding = []
    #     hc_all_node_weight = []
    #     dis_all_node_weight = []
    #     y_dis_all = []
    #     y_hc_all = []
    #     v_p_hc = {}
    #     v_p_dis = {}
    #     v_p_hc_rsa = {}
    #     v_p_dis_rsa = {}
    #     ########all cv
    #     for index in kf.split(ad_data):
    #         i = i + 1
    #             ############### Define Graph Deep Learning Network ##########################
    #         if opt.build_net:
    #             model = ContrativeNet_infomax(opt).to(device)
    #         if opt.retrain:
    #             checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
    #             model_dict = model.state_dict()
    #             pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        
    #             model_dict.update(pretrained_dict)
    #             model.load_state_dict(model_dict)
        
    #         print(model)
    #         ##############################################################           
    
    #         if opt.optimizer == 'Adam':
    #             optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
    #         elif opt.optimizer == 'SGD':
    #             optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)
            
    #         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
                        
    #         if opt.retrain:
    #             # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    #             scheduler.load_state_dict(checkpoint['scheduler'])  # 加载优化器参数
                
    #         patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = False)
    #         model.eval()
    #         for data in patient_loader:
    #             data = data.to(device)
    #             disoder_net = model.disorder_net
    #             dis_node_weight = disoder_net.target_predict.weight.detach().cpu().numpy().squeeze()
    #             output_dis, h_dis, y_dis = disoder_net(data.x,
    #                 data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             hc_net = model.HC_net    
    #             output_hc, h_hc, y_pred_hc = hc_net(data.x,
    #                   data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             y_dis = y_dis.detach().cpu().numpy().squeeze()
    #             y_pred_hc = y_pred_hc.detach().cpu().numpy().squeeze()
    #             h_dis = h_dis.detach().cpu().numpy().squeeze()
    #             h_hc = h_hc.detach().cpu().numpy().squeeze()
    #         y_dis_all.append(y_dis[dx_used == 'CN'])
    #         y_hc_all.append(y_pred_hc[dx_used == 'CN'])
    #         dis_all_node_embedding.append(h_dis[dx_used == 'CN'])
    #         hc_all_node_embedding.append(h_hc[dx_used == 'CN'])
    #     y_dis_all = np.array(y_dis_all)
    #     y_hc_all = np.array(y_hc_all)
    #     dis_all_node_embedding = np.array(dis_all_node_embedding)
    #     hc_all_node_embedding = np.array(hc_all_node_embedding)
    #     y_dis_mean = y_dis_all.mean(0)
    #     y_hc_mean = y_hc_all.mean(0)
    #     if opt.task == 'classification':
    #         y_dis_mean = y_dis_mean[:,0]
    #         y_hc_mean = y_hc_mean[:,0]
    #     y_dis_mean_all.append(y_dis_all)
    #     y_hc_mean_all.append(y_hc_all)
    #     dis_latent_all.append(dis_all_node_embedding)
    #     hc_latent_all.append(hc_all_node_embedding)
    #     hc_latent_mean = hc_all_node_embedding.mean(0)
    #     dis_latent_mean = dis_all_node_embedding.mean(0)
    #     # hc_latent_rep = cdist(hc_latent_mean, hc_latent_mean, 'correlation')
    #     for j in range(pcd_all_used.shape[-1]):
    #         # j = 7
    #         mask = ~pcd_all_used.iloc[:,j].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         y_dis = y_dis_mean[mask]
    #         y_hc = y_hc_mean[mask]
    #         hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
    #         dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
    #         hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #         dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #         if j == 0 or (mask).sum()<15:
    #             continue
    #         else:
    #             if keys_raw[j] == 'CDRSB':
    #                 feature = pcd_all_used.iloc[:,j]
    #                 feature[~mask] = 0
    #                 out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #                 out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #                 method = 0
    #                 size = len(y_dis_mean)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
                    
    #             elif keys_raw[j] == 'PTRACCAT':
    #                 mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #                 feature[feature == 'White'] = '0'
    #                 feature[feature == 'Black'] = '1'
    #                 feature[feature == 'Asian'] = '2'
    #                 feature[feature == 'More than one'] = '3'
    #                 feature = feature.astype(float)
    #                 out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #                 out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
    #             elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #                 or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #                 out_dis = spearmanr(feature, y_dis)
    #                 out_hc = spearmanr(feature, y_hc)
    #                 method = 1
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #             elif keys_raw[j] == 'APOE':
    #                 count2 = feature.str.count('2')
    #                 mask2 = count2 == 0
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #                 count3 = feature.str.count('3')
    #                 count4 = feature.str.count('4')
                    # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #                 apoe_risk = count3 * 0 + count4 * 1
    #                 apoe_risk = count4
    #                 out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #                 out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #             elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #                 or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #                 or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #                 or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for k in range(len(values)):
    #                         feature_new[feature == values[k]] = k
    #                 else:
    #                     feature_new = feature
                        
    #                 if len(values) == 2:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #                 elif len(values) == 3:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                         y_dis[feature==values[2]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                       y_hc[feature==values[2]])  
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

    #             else:
    #                 out_dis = pearsonr(feature, y_dis)
    #                 out_hc = pearsonr(feature, y_hc)
    #                 method = 2
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #         out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #         np.fill_diagonal(hc_latent_rep_cos, 0)
    #         np.fill_diagonal(dis_latent_rep_cos, 0)
    #         out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #         out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)

    #         v_p_hc[keys_raw[j]] = [out_hc[0], out_hc[1], method, size]
    #         v_p_dis[keys_raw[j]] = [out_dis[0], out_dis[1], method, size]     
    #         v_p_hc_rsa[keys_raw[j]] = [out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]]
    #         v_p_dis_rsa[keys_raw[j]] = [out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]]     
    #     v_p_all_hc.append(v_p_hc)
    #     v_p_all_dis.append(v_p_dis) 
    #     v_p_all_hc_rsa.append(v_p_hc_rsa)
    #     v_p_all_dis_rsa.append(v_p_dis_rsa)
    # v_p_hc_array = np.zeros((10, 47, 8))
    # v_p_dis_array = np.zeros((10, 47, 8))
    # keys = list(v_p_all_hc[1].keys())
    # for j in range(len(v_p_all_hc)):
    #     for k in range(len(v_p_all_hc[1])):
    #         v_p_hc_array[j, k, :4] = v_p_all_hc[j][keys[k]] 
    #         v_p_dis_array[j, k, :4] = v_p_all_dis[j][keys[k]] 
    #         v_p_hc_array[j, k, 4:] = v_p_all_hc_rsa[j][keys[k]] 
    #         v_p_dis_array[j, k, 4:] = v_p_all_dis_rsa[j][keys[k]] 
            
    # y_dis_mean_all = np.array(y_dis_mean_all)
    # y_hc_mean_all = np.array(y_hc_mean_all)
    # y_dis_mean_f = y_dis_mean_all.mean(0)
    # y_hc_mean_f = y_hc_mean_all.mean(0)
    # dis_latent_all = np.array(dis_latent_all)
    # hc_latent_all = np.array(hc_latent_all)
    # v_p_dis_f = []
    # v_p_hc_f = []
    # v_p_dis_f_rsa = []
    # v_p_hc_f_rsa = []
    # ########## association of averaged model outcomes
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     feature = pcd_all_used.iloc[:,j][mask]
    #     y_dis = y_dis_mean[mask]
    #     y_hc = y_hc_mean[mask]
    #     hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #     dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #     hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
    #     dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
        
    #     if j == 0 or (mask).sum()<15:
    #         continue
    #     else:
    #         if keys_raw[j] == 'CDRSB':
    #             feature = pcd_all_used.iloc[:,j]
    #             feature[~mask] = 0
    #             out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #             out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #             method = 0
    #             size = len(y_dis_mean)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         elif keys_raw[j] == 'PTRACCAT':
    #             mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #             feature[feature == 'White'] = '0'
    #             feature[feature == 'Black'] = '1'
    #             feature[feature == 'Asian'] = '2'
    #             feature[feature == 'More than one'] = '3'
    #             feature = feature.astype(float)
    #             out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #             out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
                    
    #         elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #             or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #             out_dis = spearmanr(feature, y_dis)
    #             out_hc = spearmanr(feature, y_hc)
    #             method = 1
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         elif keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             mask2 = count2 == 0
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #             apoe_risk = count3 * 0 + count4 * 1
    #             apoe_risk = count4
    #             out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #             out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #             # count2 = feature.str.count('2')
    #             # count3 = feature.str.count('3')
    #             # count4 = feature.str.count('4')
    #             # # apoe_risk = 0 * count2 + count3 * 1 + count4 *2
    #             # apoe_risk = count4
    #             # out_dis = f_oneway(y_dis[apoe_risk>0], y_dis[apoe_risk<=0])
    #             # out_hc = f_oneway(y_hc[apoe_risk>0], y_hc[apoe_risk<=0])
    #             # method = 0
    #             # size = len(y_dis)
    #             # feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             # feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #         elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #             or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #             or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #             or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #             values = pd.unique(feature)
    #             feature_new = np.zeros((len(feature)))
    #             if isinstance(values[0], str):
    #                 for k in range(len(values)):
    #                     feature_new[feature == values[k]] = k
    #             else:
    #                 feature_new = feature
    #             if len(values) == 2:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #             elif len(values) == 3:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                     y_dis[feature==values[2]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                   y_hc[feature==values[2]])  
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

    #         else:
    #             out_dis = pearsonr(feature, y_dis)
    #             out_hc = pearsonr(feature, y_hc)
    #             method = 2
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #     # weight_half_dis = keep_triangle_half(dis_latent_rep.shape[0]*(dis_latent_rep.shape[0]-1)//2, 1, np.expand_dims(dis_latent_rep, 0)).squeeze()
    #     # weight_half_fea = keep_triangle_half(feature_rep.shape[0]*(feature_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     # weight_half = keep_triangle_half(hc_latent_rep.shape[0]*(hc_latent_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     np.fill_diagonal(hc_latent_rep_cos, 0)
    #     np.fill_diagonal(dis_latent_rep_cos, 0)
    #     out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #     out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #     out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #     out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)
    #     v_p_hc_f.append([out_hc[0], out_hc[1], method, size])
    #     v_p_dis_f.append([out_dis[0], out_dis[1], method, size])
    #     v_p_hc_f_rsa.append([out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]])
    #     v_p_dis_f_rsa.append([out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]])

    # v_p_hc_f = np.array(v_p_hc_f)
    # v_p_dis_f = np.array(v_p_dis_f)
    # v_p_hc_f_rsa = np.array(v_p_hc_f_rsa)
    # v_p_dis_f_rsa = np.array(v_p_dis_f_rsa)
    # v_p_hc_f_ = np.delete(v_p_hc_f, [7,8,9],0) #for prediction analysis
    # v_p_dis_f_ = np.delete(v_p_dis_f, [7,8,9],0)
    # v_p_hc_f_rsa_ = np.delete(v_p_hc_f_rsa, [7,8,9],0)
    # v_p_dis_f_rsa_ = np.delete(v_p_dis_f_rsa, [7,8,9],0)
    # keys_fdr = np.delete(np.array(keys), [7,8,9],0)
    # ########## association of averaged model outcomes
    # r_p_pcd_all = []
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     r_p_pcd = []
    #     for k in range(1):
    #         # mask2 = ~pcd_all_used.iloc[:,k+1].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         feature2 = data.pcd[:,0].detach().cpu().numpy().squeeze()[dx_used == 'CN'][mask]
            
    #         if keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             feature = -1 * count2 + count3 * 0 + count4
    #         elif keys_raw[j] == 'CDRSB':
    #             feature[feature !=0] = 1
    #             print(feature)
    #             print(j)
    #         else:
    #             if isinstance(feature.iloc[0],str):
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for l in range(len(values)):
    #                         feature_new[feature == values[l]] = l
    #                 feature = feature_new
    #         size = len(feature)
    #         out = stats.pointbiserialr(feature, feature2)
    #         r_p_pcd.append([out[0], out[1], size])
    #     r_p_pcd_all.append(r_p_pcd)

    # r_p_pcd_all = np.array(r_p_pcd_all)
    # sio.savemat('/home/alex/project/CGCN/A+/final_figure/ab/statistics_withPCDassociation_withrace_NCA+.mat', {'dis_mean_final_stats': v_p_dis_f, 'hc_mean_final_stats': v_p_hc_f,
    #           'keys_all': keys, 'dis_all_final_stats': v_p_dis_array, 'hc_all_final_stats': v_p_hc_array, 'y_dis_ten_trials': y_dis_mean_all, 
    #           'y_hc_ten_trials': y_hc_mean_all, 'dis_latent_all': dis_latent_all, 'hc_latent_all': hc_latent_all, 'dis_latent_features': dis_latent_all,
    #           'hc_latent_features': hc_latent_all, 'hc_mean_final_stats_rsa': v_p_hc_f_rsa, 'dis_mean_final_stats_rsa': v_p_dis_f_rsa, 
    #           'hc_notarget_stats': v_p_hc_f_, 'dis_notarget_stats': v_p_dis_f_, 'hc_notarget_stats_rsa': v_p_hc_f_rsa_, 'dis_notarget_stats_rsa':
    #               v_p_dis_f_rsa_, 'keys_notarget': keys_fdr, 'r_p_pcd': r_p_pcd_all})
    # pcd_all_used.to_csv('/home/alex/project/CGCN/A+/final_figure/ab/pcd_feature_withrace_NCA+.csv')

    ###################################################
    ###################################RSA A+ MCI
    ##############################################
    # device = torch.device("cuda:0")
    # # device = torch.device("cpu")
    # with_contrast = True
    # hc_all_node_embeddings, dis_all_node_embeddings, hc_all_node_weights, dis_all_node_weights = [], [], [], []
    # trials = [1,11,2,14,0,8,7,6,19,5]
    # # trials = range(10)
    # v_p_all_hc = []
    # v_p_all_dis = []
    # v_p_all_hc_rsa = []
    # v_p_all_dis_rsa = []
    # y_dis_mean_all = []
    # y_hc_mean_all = []
    # dis_latent_all = []
    # hc_latent_all = []
    # for trial in trials:
    #     parser = argparse.ArgumentParser()
    #     if with_contrast:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/model/seed2/0.2/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/result/seed2/0.2/{}'.format(trial), 
    #                             help='result save path')
    #     else:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/model/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/result/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='result save path')
    #     parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs of training')
    #     parser.add_argument('--batchSize', type=int, default= 700, help='size of the batches')
    #     parser.add_argument('--fold', type=int, default=5, help='training which fold')
    #     parser.add_argument('--lr', type = float, default=0.1, help='learning rate')
    #     parser.add_argument('--stepsize', type=int, default=250, help='scheduler step size')
    #     # parser.add_argument('--stepsize', type=int, default=22, help='scheduler step size')
    #     parser.add_argument('--weightdecay', type=float, default=0.01, help='regularization')
    #     # parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
    #     parser.add_argument('--gamma', type=float, default=0.4, help='scheduler shrinking rate')
    #     parser.add_argument('--alpha', type=float, default=1, help='loss control to disentangle HC and disorder')
    #     parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
    #     parser.add_argument('--beta', type=float, default=1, help='loss control to force gaussian distribution')
    #     parser.add_argument('--theta1', type=float, default=1, help='loss control to prediction task for dis')
    #     parser.add_argument('--theta2', type=float, default=0.2, help='loss control to prediction task for HC')
    #     parser.add_argument('--build_net', default=True, type=bool, help='model name')
    #     parser.add_argument('--in_channels', type=int, default=100)
    #     parser.add_argument('--hidden_channels', type=int, default=50)
    #     parser.add_argument('--depth', type=int, default=1)
    #     parser.add_argument('--conv', type=str, default='edgeconv', help='edgeconv || gat || gcn || gen')
    #     parser.add_argument('--act', type=str, default='tanh', help='relu || leaky_relu || prelu || tanh')
    #     parser.add_argument('--sum_res', type=bool, default=True)
    #     parser.add_argument('--save_model', action='store_true')
    #     parser.add_argument('--normalization', action='store_true') 
    #     parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
    #     parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    #     parser.add_argument('--dataroot', type=str,
    #                         default=r'/home/alex/project/CGCN/A+/ptau/data/',
    #                         help='root directory of the dataset')
    #     parser.add_argument('--retrain', default=True, type=bool, help='whether train from used model')     
    #     parser.add_argument('--epsilon', default=0.1, type=float, help='stochastic epsilon for gcn')
    #     parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
    #     parser.add_argument('--demean', type=bool, default=True)
    #     parser.add_argument('--drop', default=0.3, type=float, help='drop ratio')
    #     parser.add_argument('--task', default='regression_hc_visual', type=str, help='classfication / regression/regression_hc_visual/classification_hc_visual')
    #     parser.add_argument('--augmentation', default=10, type=int, help='times of augmentation')
    #     parser.add_argument('--cluster', default=7, type=int, help='cluster number')

    #     parser.set_defaults(save_model=True)
    #     parser.set_defaults(normalization=True)
    #     opt = parser.parse_args()
    #     name = 'Biopoint'
    
    #     dataset = BiopointDataset(opt, name)
    #     HC_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0]
    #     ad_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0]
    #     hc_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),2][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0].numpy()
    #     hc_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]==0,2].numpy())
    #     hc_target_idx = np.where(~np.isnan(hc_target))[0]
    #     ad_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),3][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
    #     ad_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]!=0,2].numpy())
    #     ad_target_idx = np.where(~np.isnan(ad_target))[0]
    #     HC_data_aug = dataset[dataset.data.pcd[:,-1]==0]
    #     ad_data_aug = dataset[dataset.data.pcd[:,-1]!=0]
    #     ################################## node embedding

    #     adni_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
    #     adni_pcd_used = adni_pcd[['subjectID', 'AGE', 'gender', 'MOCA', 'ADAS11', 'ADAS13', 'PTEDUCAT', 'PTRACCAT', 'TAU', 'PTAU', 
    #                               'ABETA', 'CDRSB', 'MMSE', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
    #                               'EcogPtTotal', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting']]
    #     adni_pcd_used['Systolic_blood_pressure'] = np.nan
    #     adni_pcd_used['Diastolic_blood_pressure'] = np.nan
    #     adni_dx = adni_pcd['DX']
    #     adni_pcd_used['subjectID'] = adni_pcd_used['subjectID'].str[6:].astype(int)
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Male'] = 'M'
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Female'] = 'F'
    #     adni_bp = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/AV45VITALS_20Aug2023.csv')[['RID', 'VISCODE2', 'PRESYSTBP', 'PREDIABP']]
    #     adni_bp_bl = adni_bp[adni_bp['VISCODE2'] == 'bl']
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_bp_bl['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-2] = adni_bp_bl.iloc[IB]['PRESYSTBP']
    #     adni_pcd_used.iloc[IA,-1] = adni_bp_bl.iloc[IB]['PREDIABP'] #no IL factors accessible
        
    #     adni_apoe = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/APOERES_20Aug2023.csv')[['RID', 'APGEN1', 'APGEN2']]
    #     adni_pcd_used['APOE'] = np.nan
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_apoe['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-1] = adni_apoe.iloc[IB]['APGEN1'].astype('str') + ' ' + adni_apoe.iloc[IB]['APGEN2'].astype('str')

    #     pcd_all = adni_pcd_used
    #     dx_all = adni_dx.values
    #     pcd_mri = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),:][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
        
    #     sub, IA, IB = np.intersect1d(pcd_mri[:,0], pcd_all['subjectID'], return_indices=True)
    #     _, IAA, IBB = np.unique(IA, return_index=True, return_inverse=True)
    #     pcd_all_used = pcd_all.iloc[IB].iloc[IAA].reset_index(drop=True)
    #     dx_used = dx_all[IB][IAA]
    #     pcd_all_used = pcd_all_used[dx_used == 'MCI']
    #     keys_raw = pcd_all_used.columns.values

    #     scaler = StandardScaler()
    #     if torch.cuda.is_available():
    #         setup_seed(0) 
    #     kf = KFold(n_splits = opt.fold, shuffle=True)
    #     i = 0
    #     hc_all_node_embedding = []
    #     dis_all_node_embedding = []
    #     hc_all_node_weight = []
    #     dis_all_node_weight = []
    #     y_dis_all = []
    #     y_hc_all = []
    #     v_p_hc = {}
    #     v_p_dis = {}
    #     v_p_hc_rsa = {}
    #     v_p_dis_rsa = {}
    #     ########all cv
    #     for index in kf.split(ad_data):
    #         i = i + 1
    #             ############### Define Graph Deep Learning Network ##########################
    #         if opt.build_net:
    #             model = ContrativeNet_infomax(opt).to(device)
    #         if opt.retrain:
    #             checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
    #             model_dict = model.state_dict()
    #             pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        
    #             model_dict.update(pretrained_dict)
    #             model.load_state_dict(model_dict)
        
    #         print(model)
    #         ##############################################################           
    
    #         if opt.optimizer == 'Adam':
    #             optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
    #         elif opt.optimizer == 'SGD':
    #             optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)
            
    #         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
                        
    #         if opt.retrain:
    #             # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    #             scheduler.load_state_dict(checkpoint['scheduler'])  # 加载优化器参数
                
    #         patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = False)
    #         model.eval()
    #         for data in patient_loader:
    #             data = data.to(device)
    #             disoder_net = model.disorder_net
    #             dis_node_weight = disoder_net.target_predict.weight.detach().cpu().numpy().squeeze()
    #             output_dis, h_dis, y_dis = disoder_net(data.x,
    #                 data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             hc_net = model.HC_net    
    #             output_hc, h_hc, y_pred_hc = hc_net(data.x,
    #                   data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             y_dis = y_dis.detach().cpu().numpy().squeeze()
    #             y_pred_hc = y_pred_hc.detach().cpu().numpy().squeeze()
    #             h_dis = h_dis.detach().cpu().numpy().squeeze()
    #             h_hc = h_hc.detach().cpu().numpy().squeeze()
    #         y_dis_all.append(y_dis[IA][dx_used == 'MCI'])
    #         y_hc_all.append(y_pred_hc[IA][dx_used == 'MCI'])
    #         dis_all_node_embedding.append(h_dis[IA][dx_used == 'MCI'])
    #         hc_all_node_embedding.append(h_hc[IA][dx_used == 'MCI'])
    #     y_dis_all = np.array(y_dis_all)
    #     y_hc_all = np.array(y_hc_all)
    #     dis_all_node_embedding = np.array(dis_all_node_embedding)
    #     hc_all_node_embedding = np.array(hc_all_node_embedding)
    #     y_dis_mean = y_dis_all.mean(0)
    #     y_hc_mean = y_hc_all.mean(0)
    #     if opt.task == 'classification':
    #         y_dis_mean = y_dis_mean[:,0]
    #         y_hc_mean = y_hc_mean[:,0]
    #     y_dis_mean_all.append(y_dis_all)
    #     y_hc_mean_all.append(y_hc_all)
    #     dis_latent_all.append(dis_all_node_embedding)
    #     hc_latent_all.append(hc_all_node_embedding)
    #     hc_latent_mean = hc_all_node_embedding.mean(0)
    #     dis_latent_mean = dis_all_node_embedding.mean(0)
    #     # hc_latent_rep = cdist(hc_latent_mean, hc_latent_mean, 'correlation')
    #     for j in range(pcd_all_used.shape[-1]):
    #         # j = 7
    #         mask = ~pcd_all_used.iloc[:,j].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         y_dis = y_dis_mean[mask]
    #         y_hc = y_hc_mean[mask]
    #         hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
    #         dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
    #         hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #         dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #         if j == 0 or (mask).sum()<15:
    #             continue
    #         else:
    #             if keys_raw[j] == 'CDRSB':
    #                 feature = pcd_all_used.iloc[:,j]
    #                 feature[~mask] = 0
    #                 out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #                 out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #                 method = 0
    #                 size = len(y_dis_mean)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
                    
    #             elif keys_raw[j] == 'PTRACCAT':
    #                 mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #                 feature[feature == 'White'] = '0'
    #                 feature[feature == 'Black'] = '1'
    #                 feature[feature == 'Asian'] = '2'
    #                 feature[feature == 'More than one'] = '3'
    #                 feature = feature.astype(float)
    #                 out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #                 out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
    #             elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #                 or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #                 out_dis = spearmanr(feature, y_dis)
    #                 out_hc = spearmanr(feature, y_hc)
    #                 method = 1
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #             elif keys_raw[j] == 'APOE':
    #                 count2 = feature.str.count('2')
    #                 mask2 = count2 == 0
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #                 count3 = feature.str.count('3')
    #                 count4 = feature.str.count('4')
    #                 # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #                 apoe_risk = count3 * 0 + count4 * 1
    #                 apoe_risk = count4
    #                 out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #                 out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #             elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #                 or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #                 or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #                 or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for k in range(len(values)):
    #                         feature_new[feature == values[k]] = k
    #                 else:
    #                     feature_new = feature
                        
    #                 if len(values) == 2:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #                 elif len(values) == 3:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                         y_dis[feature==values[2]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                       y_hc[feature==values[2]])  
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

    #             else:
    #                 out_dis = pearsonr(feature, y_dis)
    #                 out_hc = pearsonr(feature, y_hc)
    #                 method = 2
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #         out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #         np.fill_diagonal(hc_latent_rep_cos, 0)
    #         np.fill_diagonal(dis_latent_rep_cos, 0)
    #         out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #         out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)

    #         v_p_hc[keys_raw[j]] = [out_hc[0], out_hc[1], method, size]
    #         v_p_dis[keys_raw[j]] = [out_dis[0], out_dis[1], method, size]     
    #         v_p_hc_rsa[keys_raw[j]] = [out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]]
    #         v_p_dis_rsa[keys_raw[j]] = [out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]]     
    #     v_p_all_hc.append(v_p_hc)
    #     v_p_all_dis.append(v_p_dis) 
    #     v_p_all_hc_rsa.append(v_p_hc_rsa)
    #     v_p_all_dis_rsa.append(v_p_dis_rsa)
    # v_p_hc_array = np.zeros((10, 47, 8))
    # v_p_dis_array = np.zeros((10, 47, 8))
    # keys = list(v_p_all_hc[1].keys())
    # for j in range(len(v_p_all_hc)):
    #     for k in range(len(v_p_all_hc[1])):
    #         v_p_hc_array[j, k, :4] = v_p_all_hc[j][keys[k]] 
    #         v_p_dis_array[j, k, :4] = v_p_all_dis[j][keys[k]] 
    #         v_p_hc_array[j, k, 4:] = v_p_all_hc_rsa[j][keys[k]] 
    #         v_p_dis_array[j, k, 4:] = v_p_all_dis_rsa[j][keys[k]] 
            
    # y_dis_mean_all = np.array(y_dis_mean_all)
    # y_hc_mean_all = np.array(y_hc_mean_all)
    # y_dis_mean_f = y_dis_mean_all.mean(0)
    # y_hc_mean_f = y_hc_mean_all.mean(0)
    # dis_latent_all = np.array(dis_latent_all)
    # hc_latent_all = np.array(hc_latent_all)
    # v_p_dis_f = []
    # v_p_hc_f = []
    # v_p_dis_f_rsa = []
    # v_p_hc_f_rsa = []
    # ########## association of averaged model outcomes
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     feature = pcd_all_used.iloc[:,j][mask]
    #     y_dis = y_dis_mean[mask]
    #     y_hc = y_hc_mean[mask]
    #     hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #     dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #     hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
    #     dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
        
    #     if j == 0 or (mask).sum()<15:
    #         continue
    #     else:
    #         if keys_raw[j] == 'CDRSB':
    #             feature = pcd_all_used.iloc[:,j]
    #             feature[~mask] = 0
    #             out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #             out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #             method = 0
    #             size = len(y_dis_mean)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         elif keys_raw[j] == 'PTRACCAT':
    #             mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #             feature[feature == 'White'] = '0'
    #             feature[feature == 'Black'] = '1'
    #             feature[feature == 'Asian'] = '2'
    #             feature[feature == 'More than one'] = '3'
    #             feature = feature.astype(float)
    #             out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #             out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
                    
    #         elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #             or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #             out_dis = spearmanr(feature, y_dis)
    #             out_hc = spearmanr(feature, y_hc)
    #             method = 1
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         elif keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             mask2 = count2 == 0
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #             apoe_risk = count3 * 0 + count4 * 1
    #             apoe_risk = count4
    #             out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #             out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #             # count2 = feature.str.count('2')
    #             # count3 = feature.str.count('3')
    #             # count4 = feature.str.count('4')
    #             # # apoe_risk = 0 * count2 + count3 * 1 + count4 *2
    #             # apoe_risk = count4
    #             # out_dis = f_oneway(y_dis[apoe_risk>0], y_dis[apoe_risk<=0])
    #             # out_hc = f_oneway(y_hc[apoe_risk>0], y_hc[apoe_risk<=0])
    #             # method = 0
    #             # size = len(y_dis)
    #             # feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             # feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #         elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #             or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #             or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #             or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #             values = pd.unique(feature)
    #             feature_new = np.zeros((len(feature)))
    #             if isinstance(values[0], str):
    #                 for k in range(len(values)):
    #                     feature_new[feature == values[k]] = k
    #             else:
    #                 feature_new = feature
    #             if len(values) == 2:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #             elif len(values) == 3:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                     y_dis[feature==values[2]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                   y_hc[feature==values[2]])  
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

    #         else:
    #             out_dis = pearsonr(feature, y_dis)
    #             out_hc = pearsonr(feature, y_hc)
    #             method = 2
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #     # weight_half_dis = keep_triangle_half(dis_latent_rep.shape[0]*(dis_latent_rep.shape[0]-1)//2, 1, np.expand_dims(dis_latent_rep, 0)).squeeze()
    #     # weight_half_fea = keep_triangle_half(feature_rep.shape[0]*(feature_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     # weight_half = keep_triangle_half(hc_latent_rep.shape[0]*(hc_latent_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     np.fill_diagonal(hc_latent_rep_cos, 0)
    #     np.fill_diagonal(dis_latent_rep_cos, 0)
    #     out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #     out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #     out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #     out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)
    #     v_p_hc_f.append([out_hc[0], out_hc[1], method, size])
    #     v_p_dis_f.append([out_dis[0], out_dis[1], method, size])
    #     v_p_hc_f_rsa.append([out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]])
    #     v_p_dis_f_rsa.append([out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]])

    # v_p_hc_f = np.array(v_p_hc_f)
    # v_p_dis_f = np.array(v_p_dis_f)
    # v_p_hc_f_rsa = np.array(v_p_hc_f_rsa)
    # v_p_dis_f_rsa = np.array(v_p_dis_f_rsa)
    # v_p_hc_f_ = np.delete(v_p_hc_f, [7,8,9],0) #for prediction analysis
    # v_p_dis_f_ = np.delete(v_p_dis_f, [7,8,9],0)
    # v_p_hc_f_rsa_ = np.delete(v_p_hc_f_rsa, [7,8,9],0)
    # v_p_dis_f_rsa_ = np.delete(v_p_dis_f_rsa, [7,8,9],0)
    # keys_fdr = np.delete(np.array(keys), [7,8,9],0)
    # ########## association of averaged model outcomes
    # r_p_pcd_all = []
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     r_p_pcd = []
    #     for k in range(1):
    #         # mask2 = ~pcd_all_used.iloc[:,k+1].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         feature2 = data.pcd[:,0].detach().cpu().numpy().squeeze()[IA][dx_used == 'MCI'][mask]
            
    #         if keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             feature = -1 * count2 + count3 * 0 + count4
    #         elif keys_raw[j] == 'CDRSB':
    #             feature[feature !=0] = 1
    #             print(feature)
    #             print(j)
    #         else:
    #             if isinstance(feature.iloc[0],str):
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for l in range(len(values)):
    #                         feature_new[feature == values[l]] = l
    #                 feature = feature_new
    #         size = len(feature)
    #         out = stats.pointbiserialr(feature, feature2)
    #         r_p_pcd.append([out[0], out[1], size])
    #     r_p_pcd_all.append(r_p_pcd)

    # r_p_pcd_all = np.array(r_p_pcd_all)
    # sio.savemat('/home/alex/project/CGCN/A+/final_figure/ab/statistics_withPCDassociation_withrace_mciA+.mat', {'dis_mean_final_stats': v_p_dis_f, 'hc_mean_final_stats': v_p_hc_f,
    #           'keys_all': keys, 'dis_all_final_stats': v_p_dis_array, 'hc_all_final_stats': v_p_hc_array, 'y_dis_ten_trials': y_dis_mean_all, 
    #           'y_hc_ten_trials': y_hc_mean_all, 'dis_latent_all': dis_latent_all, 'hc_latent_all': hc_latent_all, 'dis_latent_features': dis_latent_all,
    #           'hc_latent_features': hc_latent_all, 'hc_mean_final_stats_rsa': v_p_hc_f_rsa, 'dis_mean_final_stats_rsa': v_p_dis_f_rsa, 
    #           'hc_notarget_stats': v_p_hc_f_, 'dis_notarget_stats': v_p_dis_f_, 'hc_notarget_stats_rsa': v_p_hc_f_rsa_, 'dis_notarget_stats_rsa':
    #               v_p_dis_f_rsa_, 'keys_notarget': keys_fdr, 'r_p_pcd': r_p_pcd_all})
    # pcd_all_used.to_csv('/home/alex/project/CGCN/A+/final_figure/ab/pcd_feature_withrace_mciA+.csv')
    
    
    # ###################################################
    # ###################################RSA A+ dementia
    # ##############################################
    # device = torch.device("cuda:0")
    # # device = torch.device("cpu")
    # with_contrast = True
    # hc_all_node_embeddings, dis_all_node_embeddings, hc_all_node_weights, dis_all_node_weights = [], [], [], []
    # trials = [1,11,2,14,0,8,7,6,19,5]
    # # trials = range(10)
    # v_p_all_hc = []
    # v_p_all_dis = []
    # v_p_all_hc_rsa = []
    # v_p_all_dis_rsa = []
    # y_dis_mean_all = []
    # y_hc_mean_all = []
    # dis_latent_all = []
    # hc_latent_all = []
    # for trial in trials:
    #     parser = argparse.ArgumentParser()
    #     if with_contrast:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/model/seed2/ablation/no_contrast/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/result/seed2/ablation/no_contrast/{}'.format(trial), 
    #                             help='result save path')
    #     else:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/model/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/result/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='result save path')
    #     parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs of training')
    #     parser.add_argument('--batchSize', type=int, default= 700, help='size of the batches')
    #     parser.add_argument('--fold', type=int, default=5, help='training which fold')
    #     parser.add_argument('--lr', type = float, default=0.1, help='learning rate')
    #     parser.add_argument('--stepsize', type=int, default=250, help='scheduler step size')
    #     # parser.add_argument('--stepsize', type=int, default=22, help='scheduler step size')
    #     parser.add_argument('--weightdecay', type=float, default=0.01, help='regularization')
    #     # parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
    #     parser.add_argument('--gamma', type=float, default=0.4, help='scheduler shrinking rate')
    #     parser.add_argument('--alpha', type=float, default=1, help='loss control to disentangle HC and disorder')
    #     parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
    #     parser.add_argument('--beta', type=float, default=1, help='loss control to force gaussian distribution')
    #     parser.add_argument('--theta1', type=float, default=1, help='loss control to prediction task for dis')
    #     parser.add_argument('--theta2', type=float, default=0.2, help='loss control to prediction task for HC')
    #     parser.add_argument('--build_net', default=True, type=bool, help='model name')
    #     parser.add_argument('--in_channels', type=int, default=100)
    #     parser.add_argument('--hidden_channels', type=int, default=50)
    #     parser.add_argument('--depth', type=int, default=1)
    #     parser.add_argument('--conv', type=str, default='edgeconv', help='edgeconv || gat || gcn || gen')
    #     parser.add_argument('--act', type=str, default='tanh', help='relu || leaky_relu || prelu || tanh')
    #     parser.add_argument('--sum_res', type=bool, default=True)
    #     parser.add_argument('--save_model', action='store_true')
    #     parser.add_argument('--normalization', action='store_true') 
    #     parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
    #     parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    #     parser.add_argument('--dataroot', type=str,
    #                         default=r'/home/alex/project/CGCN/A+/ptau/data/',
    #                         help='root directory of the dataset')
    #     parser.add_argument('--retrain', default=True, type=bool, help='whether train from used model')     
    #     parser.add_argument('--epsilon', default=0.1, type=float, help='stochastic epsilon for gcn')
    #     parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
    #     parser.add_argument('--demean', type=bool, default=True)
    #     parser.add_argument('--drop', default=0.3, type=float, help='drop ratio')
    #     parser.add_argument('--task', default='regression_hc_visual', type=str, help='classfication / regression/regression_hc_visual/classification_hc_visual')
    #     parser.add_argument('--augmentation', default=10, type=int, help='times of augmentation')
    #     parser.add_argument('--cluster', default=7, type=int, help='cluster number')

    #     parser.set_defaults(save_model=True)
    #     parser.set_defaults(normalization=True)
    #     opt = parser.parse_args()
    #     name = 'Biopoint'
    
    #     dataset = BiopointDataset(opt, name)
    #     HC_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0]
    #     ad_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0]
    #     hc_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),2][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0].numpy()
    #     hc_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]==0,2].numpy())
    #     hc_target_idx = np.where(~np.isnan(hc_target))[0]
    #     ad_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),3][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
    #     ad_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]!=0,2].numpy())
    #     ad_target_idx = np.where(~np.isnan(ad_target))[0]
    #     HC_data_aug = dataset[dataset.data.pcd[:,-1]==0]
    #     ad_data_aug = dataset[dataset.data.pcd[:,-1]!=0]
    #     ################################## node embedding

    #     adni_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
    #     adni_pcd_used = adni_pcd[['subjectID', 'AGE', 'gender', 'MOCA', 'ADAS11', 'ADAS13', 'PTEDUCAT', 'PTRACCAT', 'TAU', 'PTAU', 
    #                               'ABETA', 'CDRSB', 'MMSE', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
    #                               'EcogPtTotal', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting']]
    #     adni_pcd_used['Systolic_blood_pressure'] = np.nan
    #     adni_pcd_used['Diastolic_blood_pressure'] = np.nan
    #     adni_dx = adni_pcd['DX']
    #     adni_pcd_used['subjectID'] = adni_pcd_used['subjectID'].str[6:].astype(int)
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Male'] = 'M'
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Female'] = 'F'
    #     adni_bp = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/AV45VITALS_20Aug2023.csv')[['RID', 'VISCODE2', 'PRESYSTBP', 'PREDIABP']]
    #     adni_bp_bl = adni_bp[adni_bp['VISCODE2'] == 'bl']
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_bp_bl['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-2] = adni_bp_bl.iloc[IB]['PRESYSTBP']
    #     adni_pcd_used.iloc[IA,-1] = adni_bp_bl.iloc[IB]['PREDIABP'] #no IL factors accessible
        
    #     adni_apoe = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/APOERES_20Aug2023.csv')[['RID', 'APGEN1', 'APGEN2']]
    #     adni_pcd_used['APOE'] = np.nan
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_apoe['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-1] = adni_apoe.iloc[IB]['APGEN1'].astype('str') + ' ' + adni_apoe.iloc[IB]['APGEN2'].astype('str')

    #     pcd_all = adni_pcd_used
    #     dx_all = adni_dx.values
    #     pcd_mri = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),:][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
        
    #     sub, IA, IB = np.intersect1d(pcd_mri[:,0], pcd_all['subjectID'], return_indices=True)
    #     _, IAA, IBB = np.unique(IA, return_index=True, return_inverse=True)
    #     pcd_all_used = pcd_all.iloc[IB].iloc[IAA].reset_index(drop=True)
    #     dx_used = dx_all[IB][IAA]
    #     pcd_all_used = pcd_all_used[dx_used == 'Dementia']
    #     keys_raw = pcd_all_used.columns.values

    #     scaler = StandardScaler()
    #     if torch.cuda.is_available():
    #         setup_seed(0) 
    #     kf = KFold(n_splits = opt.fold, shuffle=True)
    #     i = 0
    #     hc_all_node_embedding = []
    #     dis_all_node_embedding = []
    #     hc_all_node_weight = []
    #     dis_all_node_weight = []
    #     y_dis_all = []
    #     y_hc_all = []
    #     v_p_hc = {}
    #     v_p_dis = {}
    #     v_p_hc_rsa = {}
    #     v_p_dis_rsa = {}
    #     ########all cv
    #     for index in kf.split(ad_data):
    #         i = i + 1
    #             ############### Define Graph Deep Learning Network ##########################
    #         if opt.build_net:
    #             model = ContrativeNet_infomax(opt).to(device)
    #         if opt.retrain:
    #             checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
    #             model_dict = model.state_dict()
    #             pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        
    #             model_dict.update(pretrained_dict)
    #             model.load_state_dict(model_dict)
        
    #         print(model)
    #         ##############################################################           
    
    #         if opt.optimizer == 'Adam':
    #             optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
    #         elif opt.optimizer == 'SGD':
    #             optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)
            
    #         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
                        
    #         if opt.retrain:
    #             # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    #             scheduler.load_state_dict(checkpoint['scheduler'])  # 加载优化器参数
                
    #         patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = False)
    #         model.eval()
    #         for data in patient_loader:
    #             data = data.to(device)
    #             disoder_net = model.disorder_net
    #             dis_node_weight = disoder_net.target_predict.weight.detach().cpu().numpy().squeeze()
    #             output_dis, h_dis, y_dis = disoder_net(data.x,
    #                 data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             hc_net = model.HC_net    
    #             output_hc, h_hc, y_pred_hc = hc_net(data.x,
    #                   data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             y_dis = y_dis.detach().cpu().numpy().squeeze()
    #             y_pred_hc = y_pred_hc.detach().cpu().numpy().squeeze()
    #             h_dis = h_dis.detach().cpu().numpy().squeeze()
    #             h_hc = h_hc.detach().cpu().numpy().squeeze()
    #         y_dis_all.append(y_dis[IA][dx_used == 'Dementia'])
    #         y_hc_all.append(y_pred_hc[IA][dx_used == 'Dementia'])
    #         dis_all_node_embedding.append(h_dis[IA][dx_used == 'Dementia'])
    #         hc_all_node_embedding.append(h_hc[IA][dx_used == 'Dementia'])
    #     y_dis_all = np.array(y_dis_all)
    #     y_hc_all = np.array(y_hc_all)
    #     dis_all_node_embedding = np.array(dis_all_node_embedding)
    #     hc_all_node_embedding = np.array(hc_all_node_embedding)
    #     y_dis_mean = y_dis_all.mean(0)
    #     y_hc_mean = y_hc_all.mean(0)
    #     if opt.task == 'classification':
    #         y_dis_mean = y_dis_mean[:,0]
    #         y_hc_mean = y_hc_mean[:,0]
    #     y_dis_mean_all.append(y_dis_all)
    #     y_hc_mean_all.append(y_hc_all)
    #     dis_latent_all.append(dis_all_node_embedding)
    #     hc_latent_all.append(hc_all_node_embedding)
    #     hc_latent_mean = hc_all_node_embedding.mean(0)
    #     dis_latent_mean = dis_all_node_embedding.mean(0)
    #     # hc_latent_rep = cdist(hc_latent_mean, hc_latent_mean, 'correlation')
    #     for j in range(pcd_all_used.shape[-1]):
    #         # j = 7
    #         mask = ~pcd_all_used.iloc[:,j].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         y_dis = y_dis_mean[mask]
    #         y_hc = y_hc_mean[mask]
    #         hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'chebyshev')
    #         dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'chebyshev')
    #         hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #         dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #         if j == 0 or (mask).sum()<15:
    #             continue
    #         else:
    #             if keys_raw[j] == 'CDRSB':
    #                 feature = pcd_all_used.iloc[:,j]
    #                 feature[~mask] = 0
    #                 out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #                 out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #                 method = 0
    #                 size = len(y_dis_mean)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'chebyshev')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'chebyshev')
                    
    #             elif keys_raw[j] == 'PTRACCAT':
    #                 mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'chebyshev')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'chebyshev')
    #                 feature[feature == 'White'] = '0'
    #                 feature[feature == 'Black'] = '1'
    #                 feature[feature == 'Asian'] = '2'
    #                 feature[feature == 'More than one'] = '3'
    #                 feature = feature.astype(float)
    #                 out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #                 out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')
    #             elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #                 or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #                 out_dis = spearmanr(feature, y_dis)
    #                 out_hc = spearmanr(feature, y_hc)
    #                 method = 1
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')

    #             elif keys_raw[j] == 'APOE':
    #                 count2 = feature.str.count('2')
    #                 mask2 = count2 == 0
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'chebyshev')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'chebyshev')
    #                 count3 = feature.str.count('3')
    #                 count4 = feature.str.count('4')
    #                 # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #                 apoe_risk = count3 * 0 + count4 * 1
    #                 apoe_risk = count4
    #                 out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #                 out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'chebyshev')

    #             elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #                 or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #                 or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #                 or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for k in range(len(values)):
    #                         feature_new[feature == values[k]] = k
    #                 else:
    #                     feature_new = feature
                        
    #                 if len(values) == 2:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #                 elif len(values) == 3:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                         y_dis[feature==values[2]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                       y_hc[feature==values[2]])  
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'chebyshev')

    #             else:
    #                 out_dis = pearsonr(feature, y_dis)
    #                 out_hc = pearsonr(feature, y_hc)
    #                 method = 2
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')

    #         out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #         out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #         np.fill_diagonal(hc_latent_rep_cos, 0)
    #         np.fill_diagonal(dis_latent_rep_cos, 0)
    #         out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #         out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)

    #         v_p_hc[keys_raw[j]] = [out_hc[0], out_hc[1], method, size]
    #         v_p_dis[keys_raw[j]] = [out_dis[0], out_dis[1], method, size]     
    #         v_p_hc_rsa[keys_raw[j]] = [out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]]
    #         v_p_dis_rsa[keys_raw[j]] = [out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]]     
    #     v_p_all_hc.append(v_p_hc)
    #     v_p_all_dis.append(v_p_dis) 
    #     v_p_all_hc_rsa.append(v_p_hc_rsa)
    #     v_p_all_dis_rsa.append(v_p_dis_rsa)
    # v_p_hc_array = np.zeros((10, 47, 8))
    # v_p_dis_array = np.zeros((10, 47, 8))
    # keys = list(v_p_all_hc[1].keys())
    # for j in range(len(v_p_all_hc)):
    #     for k in range(len(v_p_all_hc[1])):
    #         v_p_hc_array[j, k, :4] = v_p_all_hc[j][keys[k]] 
    #         v_p_dis_array[j, k, :4] = v_p_all_dis[j][keys[k]] 
    #         v_p_hc_array[j, k, 4:] = v_p_all_hc_rsa[j][keys[k]] 
    #         v_p_dis_array[j, k, 4:] = v_p_all_dis_rsa[j][keys[k]] 
            
    # y_dis_mean_all = np.array(y_dis_mean_all)
    # y_hc_mean_all = np.array(y_hc_mean_all)
    # y_dis_mean_f = y_dis_mean_all.mean(0)
    # y_hc_mean_f = y_hc_mean_all.mean(0)
    # dis_latent_all = np.array(dis_latent_all)
    # hc_latent_all = np.array(hc_latent_all)
    # v_p_dis_f = []
    # v_p_hc_f = []
    # v_p_dis_f_rsa = []
    # v_p_hc_f_rsa = []
    # ########## association of averaged model outcomes
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     feature = pcd_all_used.iloc[:,j][mask]
    #     y_dis = y_dis_mean[mask]
    #     y_hc = y_hc_mean[mask]
    #     hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #     dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #     hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'chebyshev')
    #     dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'chebyshev')
        
    #     if j == 0 or (mask).sum()<15:
    #         continue
    #     else:
    #         if keys_raw[j] == 'CDRSB':
    #             feature = pcd_all_used.iloc[:,j]
    #             feature[~mask] = 0
    #             out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #             out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #             method = 0
    #             size = len(y_dis_mean)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'chebyshev')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'chebyshev')
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')

    #         elif keys_raw[j] == 'PTRACCAT':
    #             mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'chebyshev')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'chebyshev')
    #             feature[feature == 'White'] = '0'
    #             feature[feature == 'Black'] = '1'
    #             feature[feature == 'Asian'] = '2'
    #             feature[feature == 'More than one'] = '3'
    #             feature = feature.astype(float)
    #             out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #             out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')
                    
    #         elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #             or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #             out_dis = spearmanr(feature, y_dis)
    #             out_hc = spearmanr(feature, y_hc)
    #             method = 1
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')

    #         elif keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             mask2 = count2 == 0
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'chebyshev')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'chebyshev')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #             apoe_risk = count3 * 0 + count4 * 1
    #             apoe_risk = count4
    #             out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #             out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'chebyshev')

    #             # count2 = feature.str.count('2')
    #             # count3 = feature.str.count('3')
    #             # count4 = feature.str.count('4')
    #             # # apoe_risk = 0 * count2 + count3 * 1 + count4 *2
    #             # apoe_risk = count4
    #             # out_dis = f_oneway(y_dis[apoe_risk>0], y_dis[apoe_risk<=0])
    #             # out_hc = f_oneway(y_hc[apoe_risk>0], y_hc[apoe_risk<=0])
    #             # method = 0
    #             # size = len(y_dis)
    #             # feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             # feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #         elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #             or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #             or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #             or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #             values = pd.unique(feature)
    #             feature_new = np.zeros((len(feature)))
    #             if isinstance(values[0], str):
    #                 for k in range(len(values)):
    #                     feature_new[feature == values[k]] = k
    #             else:
    #                 feature_new = feature
    #             if len(values) == 2:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #             elif len(values) == 3:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                     y_dis[feature==values[2]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                   y_hc[feature==values[2]])  
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'chebyshev')

    #         else:
    #             out_dis = pearsonr(feature, y_dis)
    #             out_hc = pearsonr(feature, y_hc)
    #             method = 2
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'chebyshev')

    #     # weight_half_dis = keep_triangle_half(dis_latent_rep.shape[0]*(dis_latent_rep.shape[0]-1)//2, 1, np.expand_dims(dis_latent_rep, 0)).squeeze()
    #     # weight_half_fea = keep_triangle_half(feature_rep.shape[0]*(feature_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     # weight_half = keep_triangle_half(hc_latent_rep.shape[0]*(hc_latent_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     np.fill_diagonal(hc_latent_rep_cos, 0)
    #     np.fill_diagonal(dis_latent_rep_cos, 0)
    #     out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #     out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #     out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #     out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)
    #     v_p_hc_f.append([out_hc[0], out_hc[1], method, size])
    #     v_p_dis_f.append([out_dis[0], out_dis[1], method, size])
    #     v_p_hc_f_rsa.append([out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]])
    #     v_p_dis_f_rsa.append([out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]])

    # v_p_hc_f = np.array(v_p_hc_f)
    # v_p_dis_f = np.array(v_p_dis_f)
    # v_p_hc_f_rsa = np.array(v_p_hc_f_rsa)
    # v_p_dis_f_rsa = np.array(v_p_dis_f_rsa)
    # v_p_hc_f_ = np.delete(v_p_hc_f, [7,8,9],0) #for prediction analysis
    # v_p_dis_f_ = np.delete(v_p_dis_f, [7,8,9],0)
    # v_p_hc_f_rsa_ = np.delete(v_p_hc_f_rsa, [7,8,9],0)
    # v_p_dis_f_rsa_ = np.delete(v_p_dis_f_rsa, [7,8,9],0)
    # keys_fdr = np.delete(np.array(keys), [7,8,9],0)
    # ########## association of averaged model outcomes
    # r_p_pcd_all = []
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     r_p_pcd = []
    #     for k in range(1):
    #         # mask2 = ~pcd_all_used.iloc[:,k+1].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         feature2 = data.pcd[:,0].detach().cpu().numpy().squeeze()[IA][dx_used == 'Dementia'][mask]
            
    #         if keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             feature = -1 * count2 + count3 * 0 + count4
    #         elif keys_raw[j] == 'CDRSB':
    #             feature[feature !=0] = 1
    #             print(feature)
    #             print(j)
    #         else:
    #             if isinstance(feature.iloc[0],str):
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for l in range(len(values)):
    #                         feature_new[feature == values[l]] = l
    #                 feature = feature_new
    #         size = len(feature)
    #         out = stats.pointbiserialr(feature, feature2)
    #         r_p_pcd.append([out[0], out[1], size])
    #     r_p_pcd_all.append(r_p_pcd)

    # r_p_pcd_all = np.array(r_p_pcd_all)
    # sio.savemat('/home/alex/project/CGCN/A+/final_figure/ab/nocontrast_model/statistics_withPCDassociation_withrace_DementiaA+.mat', {'dis_mean_final_stats': v_p_dis_f, 'hc_mean_final_stats': v_p_hc_f,
    #           'keys_all': keys, 'dis_all_final_stats': v_p_dis_array, 'hc_all_final_stats': v_p_hc_array, 'y_dis_ten_trials': y_dis_mean_all, 
    #           'y_hc_ten_trials': y_hc_mean_all, 'dis_latent_all': dis_latent_all, 'hc_latent_all': hc_latent_all, 'dis_latent_features': dis_latent_all,
    #           'hc_latent_features': hc_latent_all, 'hc_mean_final_stats_rsa': v_p_hc_f_rsa, 'dis_mean_final_stats_rsa': v_p_dis_f_rsa, 
    #           'hc_notarget_stats': v_p_hc_f_, 'dis_notarget_stats': v_p_dis_f_, 'hc_notarget_stats_rsa': v_p_hc_f_rsa_, 'dis_notarget_stats_rsa':
    #               v_p_dis_f_rsa_, 'keys_notarget': keys_fdr, 'r_p_pcd': r_p_pcd_all})
    # pcd_all_used.to_csv('/home/alex/project/CGCN/A+/final_figure/ab/nocontrast_model/pcd_feature_withrace_DementiaA+.csv')
    ###################################################
    ###################################################ras all A+ no contrast
    # ###################################################
    # device = torch.device("cuda:0")
    # # device = torch.device("cpu")
    # with_contrast = True
    # hc_all_node_embeddings, dis_all_node_embeddings, hc_all_node_weights, dis_all_node_weights = [], [], [], []
    # trials = [1,11,2,14,0,8,7,6,19,5]
    # # trials = range(10)
    # v_p_all_hc = []
    # v_p_all_dis = []
    # v_p_all_hc_rsa = []
    # v_p_all_dis_rsa = []
    # y_dis_mean_all = []
    # y_hc_mean_all = []
    # dis_latent_all = []
    # hc_latent_all = []
    # for trial in trials:
    #     parser = argparse.ArgumentParser()
    #     if with_contrast:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/model/seed2/ablation/no_contrast/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/A+/ab/result/seed2/ablation/no_contrast/{}'.format(trial), 
    #                             help='result save path')
    #     else:
    #         parser.add_argument('--model_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/model/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='model save path')
    #         parser.add_argument('--result_file', type=str, default=\
    #                             r'/home/alex/project/CGCN/ptau_code/result/ptau/A-T-N-/ensemb_non_15%_dyn_infomax_pretrain3/ablation/GAT/{}'.format(trial), 
    #                             help='result save path')
    #     parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs of training')
    #     parser.add_argument('--batchSize', type=int, default= 700, help='size of the batches')
    #     parser.add_argument('--fold', type=int, default=5, help='training which fold')
    #     parser.add_argument('--lr', type = float, default=0.1, help='learning rate')
    #     parser.add_argument('--stepsize', type=int, default=250, help='scheduler step size')
    #     # parser.add_argument('--stepsize', type=int, default=22, help='scheduler step size')
    #     parser.add_argument('--weightdecay', type=float, default=0.01, help='regularization')
    #     # parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
    #     parser.add_argument('--gamma', type=float, default=0.4, help='scheduler shrinking rate')
    #     parser.add_argument('--alpha', type=float, default=1, help='loss control to disentangle HC and disorder')
    #     parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
    #     parser.add_argument('--beta', type=float, default=1, help='loss control to force gaussian distribution')
    #     parser.add_argument('--theta1', type=float, default=1, help='loss control to prediction task for dis')
    #     parser.add_argument('--theta2', type=float, default=0.2, help='loss control to prediction task for HC')
    #     parser.add_argument('--build_net', default=True, type=bool, help='model name')
    #     parser.add_argument('--in_channels', type=int, default=100)
    #     parser.add_argument('--hidden_channels', type=int, default=50)
    #     parser.add_argument('--depth', type=int, default=1)
    #     parser.add_argument('--conv', type=str, default='edgeconv', help='edgeconv || gat || gcn || gen')
    #     parser.add_argument('--act', type=str, default='tanh', help='relu || leaky_relu || prelu || tanh')
    #     parser.add_argument('--sum_res', type=bool, default=True)
    #     parser.add_argument('--save_model', action='store_true')
    #     parser.add_argument('--normalization', action='store_true') 
    #     parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
    #     parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    #     parser.add_argument('--dataroot', type=str,
    #                         default=r'/home/alex/project/CGCN/A+/ptau/data/',
    #                         help='root directory of the dataset')
    #     parser.add_argument('--retrain', default=True, type=bool, help='whether train from used model')     
    #     parser.add_argument('--epsilon', default=0.1, type=float, help='stochastic epsilon for gcn')
    #     parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
    #     parser.add_argument('--demean', type=bool, default=True)
    #     parser.add_argument('--drop', default=0.3, type=float, help='drop ratio')
    #     parser.add_argument('--task', default='regression_hc_visual', type=str, help='classfication / regression/regression_hc_visual/classification_hc_visual')
    #     parser.add_argument('--augmentation', default=10, type=int, help='times of augmentation')
    #     parser.add_argument('--cluster', default=7, type=int, help='cluster number')

    #     parser.set_defaults(save_model=True)
    #     parser.set_defaults(normalization=True)
    #     opt = parser.parse_args()
    #     name = 'Biopoint'
    
    #     dataset = BiopointDataset(opt, name)
    #     HC_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0]
    #     ad_data = dataset[dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0]
    #     hc_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),2][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]==0].numpy()
    #     hc_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]==0,2].numpy())
    #     hc_target_idx = np.where(~np.isnan(hc_target))[0]
    #     ad_target = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),3][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
    #     ad_notarget_mask = np.isnan(dataset.data.pcd[dataset.data.pcd[:,-1]!=0,2].numpy())
    #     ad_target_idx = np.where(~np.isnan(ad_target))[0]
    #     HC_data_aug = dataset[dataset.data.pcd[:,-1]==0]
    #     ad_data_aug = dataset[dataset.data.pcd[:,-1]!=0]
    #     ################################## node embedding
    #     prevent_pcd = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
    #     prevent_pcd_used = prevent_pcd[['CONP_CandID', 'Candidate_Age', 'Gender', 'AD8_total_score', 'Systolic_blood_pressure',
    #                                     'Diastolic_blood_pressure', 'tau', 'ptau', 'Amyloid_beta_1_42', 'G_CSF', 'IL_15', 'IL_8', 'VEGF',
    #                                     'APOE', 'BchE_K_variant', 'BDNF', 'HMGCR_Intron_M', 'TLR4_rs_4986790', 'PPP2r1A_rs_10406151',
    #                                     'CDK5RAP2_rs10984186', 'immediate_memory_index_score', 'visuospatial_constructional_index_score', 
    #                                     'language_index_score', 'attention_index_score', 'delayed_memory_index_score', 'total_scale_index_score',]]
    #     prevent_demo = pd.read_csv('/home/alex/project/CGCN/dataset/AD/PREVENT-AD/PhenotypicData/Demographics_Registered_PREVENTAD.csv')[[
    #         'CONP_CandID', 'Sex', 'Ethnicity', 'Education_years', 'father_dx_ad_dementia', 'mother_dx_ad_dementia', 'sibling_dx_ad_dementia', 'other_family_members_AD']]
    #     prevent_demo = prevent_demo.rename(columns={'Sex':'Gender'})
    #     prevent_demo['Ethnicity'][prevent_demo['Ethnicity'] == 'caucasian'] = 'White'
    #     prevent_demo['Ethnicity'][prevent_demo['Ethnicity'] == 'other'] = 'Unknown'

    #     prevent_pcd_used = pd.merge(prevent_pcd_used, prevent_demo, how='outer', on=['CONP_CandID', 'Gender'])
    #     prevent_pcd_used['Candidate_Age'] = prevent_pcd_used['Candidate_Age'] / 12
    #     adni_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
    #     adni_pcd_used = adni_pcd[['subjectID', 'AGE', 'gender', 'MOCA', 'ADAS11', 'ADAS13', 'PTEDUCAT', 'PTRACCAT', 'TAU', 'PTAU', 
    #                               'ABETA', 'CDRSB', 'MMSE', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
    #                               'EcogPtTotal', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting']]
    #     adni_pcd_used['Systolic_blood_pressure'] = np.nan
    #     adni_pcd_used['Diastolic_blood_pressure'] = np.nan
    #     adni_pcd_used['subjectID'] = adni_pcd_used['subjectID'].str[6:].astype(int)
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Male'] = 'M'
    #     adni_pcd_used['gender'][adni_pcd_used['gender'] == 'Female'] = 'F'
    #     adni_bp = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/AV45VITALS_20Aug2023.csv')[['RID', 'VISCODE2', 'PRESYSTBP', 'PREDIABP']]
    #     adni_bp_bl = adni_bp[adni_bp['VISCODE2'] == 'bl']
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_bp_bl['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-2] = adni_bp_bl.iloc[IB]['PRESYSTBP']
    #     adni_pcd_used.iloc[IA,-1] = adni_bp_bl.iloc[IB]['PREDIABP'] #no IL factors accessible
        
    #     adni_apoe = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/APOERES_20Aug2023.csv')[['RID', 'APGEN1', 'APGEN2']]
    #     adni_pcd_used['APOE'] = np.nan
    #     [sub, IA, IB] = np.intersect1d(adni_pcd_used['subjectID'], adni_apoe['RID'], return_indices=True)
    #     adni_pcd_used.iloc[IA,-1] = adni_apoe.iloc[IB]['APGEN1'].astype('str') + ' ' + adni_apoe.iloc[IB]['APGEN2'].astype('str')
    #     prevent_pcd_used = prevent_pcd_used.rename(columns={'CONP_CandID': 'subjectID', 'Candidate_Age': 'AGE', 'Gender': 'gender', 'tau': 'TAU', 'ptau': 'PTAU', 
    #                                     'Amyloid_beta_1_42': 'ABETA', 'Education_years': 'PTEDUCAT', 'Ethnicity': 'PTRACCAT'})
    #     prevent_pcd_used['gender'][prevent_pcd_used['gender'] == 'Male'] = 'M'
    #     prevent_pcd_used['gender'][prevent_pcd_used['gender'] == 'Female'] = 'F'
    #     pcd_all = pd.merge(adni_pcd_used, prevent_pcd_used, how='outer', on=['subjectID', 'AGE', 'gender', 'TAU', 'PTAU', 'ABETA', 'PTEDUCAT', 'APOE', 
    #                                                                           'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'PTRACCAT'])
    #     pcd_mri = dataset.data.pcd[:len(dataset)//(opt.augmentation+1),:][dataset.data.pcd[:len(dataset)//(opt.augmentation+1),-1]!=0].numpy()
        
    #     sub, IA, IB = np.intersect1d(pcd_mri[:,0], pcd_all['subjectID'], return_indices=True)
    #     _, IAA, IBB = np.unique(IA, return_index=True, return_inverse=True)
    #     pcd_all_used = pcd_all.iloc[IB].iloc[IAA].reset_index(drop=True)
    #     keys_raw = pcd_all_used.columns.values

    #     scaler = StandardScaler()
    #     if torch.cuda.is_available():
    #         setup_seed(0) 
    #     kf = KFold(n_splits = opt.fold, shuffle=True)
    #     i = 0
    #     hc_all_node_embedding = []
    #     dis_all_node_embedding = []
    #     hc_all_node_weight = []
    #     dis_all_node_weight = []
    #     y_dis_all = []
    #     y_hc_all = []
    #     v_p_hc = {}
    #     v_p_dis = {}
    #     v_p_hc_rsa = {}
    #     v_p_dis_rsa = {}
    #     ########all cv
    #     for index in kf.split(ad_data):
    #         i = i + 1
    #             ############### Define Graph Deep Learning Network ##########################
    #         if opt.build_net:
    #             model = ContrativeNet_infomax(opt).to(device)
    #         if opt.retrain:
    #             checkpoint  = torch.load(os.path.join(opt.model_file, 'model_cv_{}.pth'.format(i)), map_location=torch.device("cuda:0"))
    #             model_dict = model.state_dict()
    #             pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        
    #             model_dict.update(pretrained_dict)
    #             model.load_state_dict(model_dict)
        
    #         print(model)
    #         ##############################################################           
    
    #         if opt.optimizer == 'Adam':
    #             optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
    #         elif opt.optimizer == 'SGD':
    #             optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)
            
    #         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
                        
    #         if opt.retrain:
    #             # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    #             scheduler.load_state_dict(checkpoint['scheduler'])  # 加载优化器参数
                
    #         patient_loader = DataLoader(ad_data,batch_size=opt.batchSize,shuffle = False)
    #         model.eval()
    #         for data in patient_loader:
    #             data = data.to(device)
    #             disoder_net = model.disorder_net
    #             dis_node_weight = disoder_net.target_predict.weight.detach().cpu().numpy().squeeze()
    #             output_dis, h_dis, y_dis = disoder_net(data.x,
    #                 data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             hc_net = model.HC_net    
    #             output_hc, h_hc, y_pred_hc = hc_net(data.x,
    #                   data.edge_index, data.edge_attr, data.pcd, data.eyes, data.batch)
    #             y_dis = y_dis.detach().cpu().numpy().squeeze()
    #             y_pred_hc = y_pred_hc.detach().cpu().numpy().squeeze()
    #             h_dis = h_dis.detach().cpu().numpy().squeeze()
    #             h_hc = h_hc.detach().cpu().numpy().squeeze()
    #         y_dis_all.append(y_dis)
    #         y_hc_all.append(y_pred_hc)
    #         dis_all_node_embedding.append(h_dis)
    #         hc_all_node_embedding.append(h_hc)
    #     y_dis_all = np.array(y_dis_all)
    #     y_hc_all = np.array(y_hc_all)
    #     dis_all_node_embedding = np.array(dis_all_node_embedding)
    #     hc_all_node_embedding = np.array(hc_all_node_embedding)
    #     y_dis_mean = y_dis_all.mean(0)
    #     y_hc_mean = y_hc_all.mean(0)
    #     if opt.task == 'classification':
    #         y_dis_mean = y_dis_mean[:,0]
    #         y_hc_mean = y_hc_mean[:,0]
    #     y_dis_mean_all.append(y_dis_all)
    #     y_hc_mean_all.append(y_hc_all)
    #     dis_latent_all.append(dis_all_node_embedding)
    #     hc_latent_all.append(hc_all_node_embedding)
    #     hc_latent_mean = hc_all_node_embedding.mean(0)
    #     dis_latent_mean = dis_all_node_embedding.mean(0)
    #     # hc_latent_rep = cdist(hc_latent_mean, hc_latent_mean, 'correlation')
    #     for j in range(pcd_all_used.shape[-1]):
    #         # j = 7
    #         mask = ~pcd_all_used.iloc[:,j].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         y_dis = y_dis_mean[mask]
    #         y_hc = y_hc_mean[mask]
    #         hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
    #         dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
    #         hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #         dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #         if j == 0 or (mask).sum()<15:
    #             continue
    #         else:
    #             if keys_raw[j] == 'CDRSB':
    #                 feature = pcd_all_used.iloc[:,j]
    #                 feature[~mask] = 0
    #                 out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #                 out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #                 method = 0
    #                 size = len(y_dis_mean)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
                    
    #             elif keys_raw[j] == 'PTRACCAT':
    #                 mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #                 feature[feature == 'White'] = '0'
    #                 feature[feature == 'Black'] = '1'
    #                 feature[feature == 'Asian'] = '2'
    #                 feature[feature == 'More than one'] = '3'
    #                 feature = feature.astype(float)
    #                 out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #                 out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
    #             elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #                 or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #                 out_dis = spearmanr(feature, y_dis)
    #                 out_hc = spearmanr(feature, y_hc)
    #                 method = 1
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #             elif keys_raw[j] == 'APOE':
    #                 count2 = feature.str.count('2')
    #                 mask2 = count2 == 0
    #                 feature = feature[mask2]
    #                 hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #                 dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #                 hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #                 dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #                 count3 = feature.str.count('3')
    #                 count4 = feature.str.count('4')
    #                 # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #                 apoe_risk = count3 * 0 + count4 * 1
    #                 apoe_risk = count4
    #                 out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #                 out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #             elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #                 or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #                 or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #                 or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for k in range(len(values)):
    #                         feature_new[feature == values[k]] = k
    #                 else:
    #                     feature_new = feature
                        
    #                 if len(values) == 2:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #                 elif len(values) == 3:
    #                     out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                         y_dis[feature==values[2]])
    #                     out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                       y_hc[feature==values[2]])  
    #                 method = 0
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

    #             else:
    #                 out_dis = pearsonr(feature, y_dis)
    #                 out_hc = pearsonr(feature, y_hc)
    #                 method = 2
    #                 size = len(y_dis)
    #                 feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #                 feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #         out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #         np.fill_diagonal(hc_latent_rep_cos, 0)
    #         np.fill_diagonal(dis_latent_rep_cos, 0)
    #         out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #         out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)

    #         v_p_hc[keys_raw[j]] = [out_hc[0], out_hc[1], method, size]
    #         v_p_dis[keys_raw[j]] = [out_dis[0], out_dis[1], method, size]     
    #         v_p_hc_rsa[keys_raw[j]] = [out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]]
    #         v_p_dis_rsa[keys_raw[j]] = [out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]]     
    #     v_p_all_hc.append(v_p_hc)
    #     v_p_all_dis.append(v_p_dis) 
    #     v_p_all_hc_rsa.append(v_p_hc_rsa)
    #     v_p_all_dis_rsa.append(v_p_dis_rsa)
    # v_p_hc_array = np.zeros((10, 47, 8))
    # v_p_dis_array = np.zeros((10, 47, 8))
    # keys = list(v_p_all_hc[1].keys())
    # for j in range(len(v_p_all_hc)):
    #     for k in range(len(v_p_all_hc[1])):
    #         v_p_hc_array[j, k, :4] = v_p_all_hc[j][keys[k]] 
    #         v_p_dis_array[j, k, :4] = v_p_all_dis[j][keys[k]] 
    #         v_p_hc_array[j, k, 4:] = v_p_all_hc_rsa[j][keys[k]] 
    #         v_p_dis_array[j, k, 4:] = v_p_all_dis_rsa[j][keys[k]] 
            
    # y_dis_mean_all = np.array(y_dis_mean_all)
    # y_hc_mean_all = np.array(y_hc_mean_all)
    # y_dis_mean_f = y_dis_mean_all.mean(0)
    # y_hc_mean_f = y_hc_mean_all.mean(0)
    # dis_latent_all = np.array(dis_latent_all)
    # hc_latent_all = np.array(hc_latent_all)
    # v_p_dis_f = []
    # v_p_hc_f = []
    # v_p_dis_f_rsa = []
    # v_p_hc_f_rsa = []
    # ########## association of averaged model outcomes
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     feature = pcd_all_used.iloc[:,j][mask]
    #     y_dis = y_dis_mean[mask]
    #     y_hc = y_hc_mean[mask]
    #     hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]))
    #     dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]))
    #     hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask]), scaler.fit_transform(hc_latent_mean[mask]), 'cosine')
    #     dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask]), scaler.fit_transform(dis_latent_mean[mask]), 'cosine')
        
    #     if j == 0 or (mask).sum()<15:
    #         continue
    #     else:
    #         if keys_raw[j] == 'CDRSB':
    #             feature = pcd_all_used.iloc[:,j]
    #             feature[~mask] = 0
    #             out_dis = f_oneway(y_dis_mean[feature==0], y_dis_mean[feature!=0])
    #             out_hc = f_oneway(y_hc_mean[feature==0], y_hc_mean[feature!=0])
    #             method = 0
    #             size = len(y_dis_mean)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean), scaler.fit_transform(hc_latent_mean), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean), scaler.fit_transform(dis_latent_mean), 'cosine')
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         elif keys_raw[j] == 'PTRACCAT':
    #             mask2 = (pcd_all_used['PTRACCAT'] != 'Unknown')
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #             feature[feature == 'White'] = '0'
    #             feature[feature == 'Black'] = '1'
    #             feature[feature == 'Asian'] = '2'
    #             feature[feature == 'More than one'] = '3'
    #             feature = feature.astype(float)
    #             out_dis = f_oneway(y_dis[mask2][feature==0], y_dis[mask2][feature!=0])
    #             out_hc = f_oneway(y_hc[mask2][feature==0], y_hc[mask2][feature!=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')
                    
    #         elif keys_raw[j] == 'EcogPtMem' or keys_raw[j] == 'EcogPtLang' or keys_raw[j] == 'EcogPtVisspat' or keys_raw[j] == 'EcogPtPlan'\
    #             or keys_raw[j] == 'EcogPtOrgan' or keys_raw[j] == 'EcogPtDivatt':
    #             out_dis = spearmanr(feature, y_dis)
    #             out_hc = spearmanr(feature, y_hc)
    #             method = 1
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #         elif keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             mask2 = count2 == 0
    #             feature = feature[mask2]
    #             hc_latent_rep = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]))
    #             dis_latent_rep = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]))
    #             hc_latent_rep_cos = cdist(scaler.fit_transform(hc_latent_mean[mask][mask2]), scaler.fit_transform(hc_latent_mean[mask][mask2]), 'cosine')
    #             dis_latent_rep_cos = cdist(scaler.fit_transform(dis_latent_mean[mask][mask2]), scaler.fit_transform(dis_latent_mean[mask][mask2]), 'cosine')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             # apoe_risk = -1 * count2 + count3 * 0 + (count4!=0) * 1
    #             apoe_risk = count3 * 0 + count4 * 1
    #             apoe_risk = count4
    #             out_dis = f_oneway(y_dis[mask2][apoe_risk>0], y_dis[mask2][apoe_risk<=0])
    #             out_hc = f_oneway(y_hc[mask2][apoe_risk>0], y_hc[mask2][apoe_risk<=0])
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #             # count2 = feature.str.count('2')
    #             # count3 = feature.str.count('3')
    #             # count4 = feature.str.count('4')
    #             # # apoe_risk = 0 * count2 + count3 * 1 + count4 *2
    #             # apoe_risk = count4
    #             # out_dis = f_oneway(y_dis[apoe_risk>0], y_dis[apoe_risk<=0])
    #             # out_hc = f_oneway(y_hc[apoe_risk>0], y_hc[apoe_risk<=0])
    #             # method = 0
    #             # size = len(y_dis)
    #             # feature_rep = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)))
    #             # feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(apoe_risk,-1)), scaler.fit_transform(np.expand_dims(apoe_risk,-1)), 'cosine')

    #         elif keys_raw[j] == 'gender' or keys_raw[j] == 'BchE_K_variant' or keys_raw[j] == 'AD8_total_score' or keys_raw[j] == 'BDNF'\
    #             or keys_raw[j] == 'HMGCR_Intron_M' or keys_raw[j] == 'TLR4_rs_4986790' or keys_raw[j] == 'PPP2r1A_rs_10406151'\
    #             or keys_raw[j] == 'CDK5RAP2_rs10984186' or keys_raw[j] == 'father_dx_ad_dementia' or keys_raw[j] == 'mother_dx_ad_dementia'\
    #             or keys_raw[j] == 'sibling_dx_ad_dementia' or keys_raw[j] == 'other_family_members_AD':
    #             values = pd.unique(feature)
    #             feature_new = np.zeros((len(feature)))
    #             if isinstance(values[0], str):
    #                 for k in range(len(values)):
    #                     feature_new[feature == values[k]] = k
    #             else:
    #                 feature_new = feature
    #             if len(values) == 2:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]])  
    #             elif len(values) == 3:
    #                 out_dis = f_oneway(y_dis[feature==values[0]], y_dis[feature==values[1]], 
    #                                     y_dis[feature==values[2]])
    #                 out_hc = f_oneway(y_hc[feature==values[0]], y_hc[feature==values[1]],
    #                                   y_hc[feature==values[2]])  
    #             method = 0
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature_new,-1)), scaler.fit_transform(np.expand_dims(feature_new,-1)), 'cosine')

    #         else:
    #             out_dis = pearsonr(feature, y_dis)
    #             out_hc = pearsonr(feature, y_hc)
    #             method = 2
    #             size = len(y_dis)
    #             feature_rep = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)))
    #             feature_rep_cos = cdist(scaler.fit_transform(np.expand_dims(feature,-1)), scaler.fit_transform(np.expand_dims(feature,-1)), 'cosine')

    #     # weight_half_dis = keep_triangle_half(dis_latent_rep.shape[0]*(dis_latent_rep.shape[0]-1)//2, 1, np.expand_dims(dis_latent_rep, 0)).squeeze()
    #     # weight_half_fea = keep_triangle_half(feature_rep.shape[0]*(feature_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     # weight_half = keep_triangle_half(hc_latent_rep.shape[0]*(hc_latent_rep.shape[0]-1)//2, 1, np.expand_dims(feature_rep, 0)).squeeze()
    #     np.fill_diagonal(hc_latent_rep_cos, 0)
    #     np.fill_diagonal(dis_latent_rep_cos, 0)
    #     out_hc_rsa = mantel(feature_rep, hc_latent_rep, 'spearman', 10000)
    #     out_dis_rsa = mantel(feature_rep, dis_latent_rep, 'spearman', 10000)
    #     out_hc_rsa_corr = mantel(feature_rep_cos, hc_latent_rep_cos, 'spearman', 10000)
    #     out_dis_rsa_corr = mantel(feature_rep_cos, dis_latent_rep_cos, 'spearman', 10000)
    #     v_p_hc_f.append([out_hc[0], out_hc[1], method, size])
    #     v_p_dis_f.append([out_dis[0], out_dis[1], method, size])
    #     v_p_hc_f_rsa.append([out_hc_rsa[0], out_hc_rsa[1], out_hc_rsa_corr[0], out_hc_rsa_corr[1]])
    #     v_p_dis_f_rsa.append([out_dis_rsa[0], out_dis_rsa[1], out_dis_rsa_corr[0], out_dis_rsa_corr[1]])

    # v_p_hc_f = np.array(v_p_hc_f)
    # v_p_dis_f = np.array(v_p_dis_f)
    # v_p_hc_f_rsa = np.array(v_p_hc_f_rsa)
    # v_p_dis_f_rsa = np.array(v_p_dis_f_rsa)
    # v_p_hc_f_ = np.delete(v_p_hc_f, [7,8,9],0) #for prediction analysis
    # v_p_dis_f_ = np.delete(v_p_dis_f, [7,8,9],0)
    # v_p_hc_f_rsa_ = np.delete(v_p_hc_f_rsa, [7,8,9],0)
    # v_p_dis_f_rsa_ = np.delete(v_p_dis_f_rsa, [7,8,9],0)
    # keys_fdr = np.delete(np.array(keys), [7,8,9],0)
    # ########## association of averaged model outcomes
    # r_p_pcd_all = []
    # for j in range(pcd_all_used.shape[-1]):
    #     mask = ~pcd_all_used.iloc[:,j].isnull()
    #     r_p_pcd = []
    #     for k in range(1):
    #         # mask2 = ~pcd_all_used.iloc[:,k+1].isnull()
    #         feature = pcd_all_used.iloc[:,j][mask]
    #         feature2 = data.pcd[:,0].detach().cpu().numpy().squeeze()[mask]
            
    #         if keys_raw[j] == 'APOE':
    #             count2 = feature.str.count('2')
    #             count3 = feature.str.count('3')
    #             count4 = feature.str.count('4')
    #             feature = -1 * count2 + count3 * 0 + count4
    #         elif keys_raw[j] == 'CDRSB':
    #             feature[feature !=0] = 1
    #             print(feature)
    #             print(j)
    #         else:
    #             if isinstance(feature.iloc[0],str):
    #                 values = pd.unique(feature)
    #                 feature_new = np.zeros((len(feature)))
    #                 if isinstance(values[0], str):
    #                     for l in range(len(values)):
    #                         feature_new[feature == values[l]] = l
    #                 feature = feature_new
    #         size = len(feature)
    #         out = stats.pointbiserialr(feature, feature2)
    #         r_p_pcd.append([out[0], out[1], size])
    #     r_p_pcd_all.append(r_p_pcd)

    # r_p_pcd_all = np.array(r_p_pcd_all)
    # sio.savemat('/home/alex/project/CGCN/A+/final_figure/ab/nocontrast_model/statistics_withPCDassociation_withrace_A+_all.mat', {'dis_mean_final_stats': v_p_dis_f, 'hc_mean_final_stats': v_p_hc_f,
    #           'keys_all': keys, 'dis_all_final_stats': v_p_dis_array, 'hc_all_final_stats': v_p_hc_array, 'y_dis_ten_trials': y_dis_mean_all, 
    #           'y_hc_ten_trials': y_hc_mean_all, 'dis_latent_all': dis_latent_all, 'hc_latent_all': hc_latent_all, 'dis_latent_features': dis_latent_all,
    #           'hc_latent_features': hc_latent_all, 'hc_mean_final_stats_rsa': v_p_hc_f_rsa, 'dis_mean_final_stats_rsa': v_p_dis_f_rsa, 
    #           'hc_notarget_stats': v_p_hc_f_, 'dis_notarget_stats': v_p_dis_f_, 'hc_notarget_stats_rsa': v_p_hc_f_rsa_, 'dis_notarget_stats_rsa':
    #               v_p_dis_f_rsa_, 'keys_notarget': keys_fdr, 'r_p_pcd': r_p_pcd_all})
    # pcd_all_used.to_csv('/home/alex/project/CGCN/A+/final_figure/ab/nocontrast_model/pcd_feature_withrace_A+_all.csv')
