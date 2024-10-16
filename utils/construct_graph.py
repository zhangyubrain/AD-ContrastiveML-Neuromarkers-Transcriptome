
import os.path as osp
from os import listdir
import pandas as pd
import os
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import scipy.io as sio
from scipy.spatial import distance
import math
import copy
from numpy.linalg import eig
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import sys
sys.path.append(r'/home/alex/project/utils_for_all')
from common_utils import get_function_connectivity, read_singal_fmri_ts, get_fc, keep_triangle_half, vector_to_matrix
from neuroCombat import neuroCombat

def get_fc_aug(ts_list, tp, aug_times = 2):
    fc_array = np.zeros((aug_times, len(ts_list),100,100))
    for time in range(aug_times):
        for i, ts in enumerate(ts_list):
                idx = np.random.randint(low = 0, high = ts.shape[1]-1, size = 1)[0]
                while idx > len(ts_list) - tp:
                    idx = np.random.randint(low = 0, high = ts.shape[1]-1, size = 1)[0]
                    break
                ts = ts[idx:idx+tp]
                fc = get_function_connectivity(ts)
                np.fill_diagonal(fc, 0)
                transf_fc = np.arctanh(fc)
                fc_array[time,i] = transf_fc

                # while True:
                #     if np.isinf(transf_fc).sum()==0:
                #         fc_array[time,i] = transf_fc
                #         break
                #     else:
                #         idx = np.random.randint(low = 0, high = ts.shape[1]-1, size = 1)[0]
                #         while idx > len(ts_list) - tp:
                #             idx = np.random.randint(low = 0, high = ts.shape[1]-1, size = 1)[0]
                #             break
                #         ts = ts[idx:idx+tp]
                #         fc = get_function_connectivity(ts)
                #         np.fill_diagonal(fc, 0)
                #         transf_fc = np.arctanh(fc)
    return fc_array

def split(data, batch1):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch1)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch1[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])  

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch1[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}   
 
    if data.x is not None:
        slices['x'] = node_slice    
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice  
    # if data.pos is not None:
    #     slices['eng_vect'] = node_slice
    if data.pos is not None:
        slices['pcd'] = torch.arange(0, batch1[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['eyes'] = node_slice
    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

def get_site_idx(info_list):
    site_list = [x[-1] for x in info_list]
    site_set = set(site_list)
    index_list = []
    for j in site_set:
        index = []
        for i,info in enumerate(site_list):
            if info == j:
                index.append(i)
        index_list.append(index)
            
    return index_list

def norm_site(node_fea, edge, index):
    node_array = np.array(node_fea)
    edge_array = np.array(edge)
    node = range(node_array.shape[1])

    for i in range(len(index)):
        min_node = np.min(node_array[index[i]],0)
        min_edge = np.min(edge_array[index[i]],0)
        max_node = np.max(node_array[index[i]],0)
        max_edge = np.max(edge_array[index[i]],0)
        avr_node = np.mean(node_array[index[i]],axis=0)
        avr_edge = np.mean(edge_array[index[i]],axis=0)
        std_node = np.std(node_array[index[i]],axis=0)
        std_edge = np.std(edge_array[index[i]],axis=0)
        #stardadization
        node_array[index[i]] = (node_array[index[i]]-avr_node)/std_node
        edge_array[index[i]] = (edge_array[index[i]]-avr_edge)/std_edge
        #norm
        range_node = (max_node-min_node) 
        node_array[index[i]] = t = (node_array[index[i]]-min_node)/range_node
        edge_array[index[i]] = (edge_array[index[i]]-min_edge)/(max_edge-min_edge)
        
    for j in range(len(node_array)):
        node_array[j][node,node] = 0
    return node_array, edge_array

def get_singal_graph(edge, node_fea, threshold):
    # only keep the top 10% edges
    th = np.percentile(abs(edge).reshape(-1),threshold)
    edge[abs(edge) < th] = 0  # set a threshold
    num_nodes = edge.shape[0]

    G = from_numpy_matrix(edge)
    # try:
    #     L = nx.average_shortest_path_length(G)
    # except Exception:
    #     print('isolate')
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros((len(adj.row)))
    for i in range(len(adj.row)):
        edge_att[i] = edge[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index).long(), torch.from_numpy(edge_att).float())
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    return edge_att.data.numpy(),edge_index.data.numpy(),node_fea, num_nodes
    
def read_data_regression_abOAS_f(opt):
    
    #############csf
    select_fc = sio.loadmat('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_fc.mat')['FC_bl']
    select_pcd_all = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_csfRExtract.csv')
    select_pcd = select_pcd_all[select_pcd_all['TimePoint'] == 'bl']
    ts_file = open(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_ts_bl.pkl', 'rb')
    select_ts = pickle.load(ts_file)
    ts_file.close()
    aug_fc = get_fc_aug(select_ts, 100, opt.augmentation)
    select_pcd_used = select_pcd[['subjectID', 'gender', 'AGE', 'TAU', 'PTAU', 'ABETA', 'DX']]
    select_pcd_used['subjectID'] = select_pcd_used['subjectID'].str[6:]

    select_fc = select_fc[~select_pcd_used['DX'].isnull()]
    aug_fc = aug_fc[:,~select_pcd_used['DX'].isnull()]
    select_pcd_used = select_pcd_used[~select_pcd_used['DX'].isnull()]

    select_pcd_used = select_pcd_used.values
    norm_mask = (select_pcd_used[:,-2] >= 977)
    select_pcd_used[~norm_mask,-1] = 'AD risk'
    select_pcd_used[norm_mask,-1] = 'CN'
    
    select_fc2 = sio.loadmat(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau3.mat')['fmri_withtau_all_bl']
    select_pcd2 = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
    select_pcd_used2 = select_pcd2[['CONP_CandID', 'Gender', 'Candidate_Age', 'tau', 'ptau', 'Amyloid_beta_1_42']]
    ts_file2 = open(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/ts_withtau_bl.pkl', 'rb')
    select_ts2 = pickle.load(ts_file2)
    ts_file2.close()
    aug_fc2 = get_fc_aug(select_ts2, 100, opt.augmentation)
    select_fc2 = select_fc2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())]
    aug_fc2 = aug_fc2[:,~(select_pcd_used2['Amyloid_beta_1_42'].isna())]
    select_pcd_used2 = select_pcd_used2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())].values

    select_dx2 = []
    for i in range(len(select_pcd_used2)):
        # if (select_pcd_used2[i, -1] >= 977 ) and (select_pcd_used2[i, -2] <= 22):
        if (select_pcd_used2[i, -1] >= 977):
            select_dx2.append('CN')
        else:
            select_dx2.append('AD risk')
    select_pcd_used2 = np.c_[select_pcd_used2, select_dx2]

    # if opt.demean:
    #     select_fc = select_fc - select_fc.mean()
    #     select_fc2 = select_fc2 - select_fc2.mean(-1).mean()

    select_fc_used = np.r_[select_fc, select_fc2]
    select_fc_used_aug = np.concatenate((aug_fc, aug_fc2), 1)

    select_fc_used = np.expand_dims(select_fc_used, 0)
    select_fc = np.r_[select_fc_used, select_fc_used_aug]
    select_fc = select_fc.reshape(-1,100,100)

    select_pcd = np.r_[select_pcd_used, select_pcd_used2]
    select_pcd[(select_pcd[:,1]=='F')|((select_pcd[:,1]=='Female')),1] = '0' 
    select_pcd[(select_pcd[:,1]=='M')|((select_pcd[:,1]=='Male')),1] = '1' 
    select_pcd[select_pcd[:,-1] == 'CN',-1] = '0'
    select_pcd[select_pcd[:,-1] == 'AD risk',-1] = '1'
    select_pcd = np.array(select_pcd).astype(float)
    select_pcd = np.tile(select_pcd,(opt.augmentation+1,1))

    #######################################pet
    oas_info = sio.loadmat('/home/alex/project/CGCN/dataset/AD/OASIS3/fc_pup_f2.mat')
    oas_info2 = sio.loadmat('/home/alex/project/CGCN/dataset/AD/OASIS3/fc_pup_f.mat')
    select_ts_oas = oas_info['ts'].squeeze()
    select_fc_oas = oas_info['fc'].squeeze()
    aug_fc_oas = get_fc_aug(select_ts_oas, 100, opt.augmentation)
    sub = np.array([i[0][3:8] for i in oas_info['sub'].squeeze()]).astype(float)
    tracer = np.array([i[0] for i in oas_info['tracer'].squeeze()])
    Centil_fBP_TOT_CORTMEAN = np.array([i[0] for i in oas_info['Centil_fBP_TOT_CORTMEAN'].squeeze()]).squeeze().astype(float)
    Centil_fSUVR_TOT_CORTMEAN = np.array([i[0] for i in oas_info['Centil_fSUVR_TOT_CORTMEAN'].squeeze()]).squeeze().astype(float)
    Centil_fBP_rsf_TOT_CORTMEAN = np.array([i[0] for i in oas_info['Centil_fBP_rsf_TOT_CORTMEAN'].squeeze()]).squeeze().astype(float)
    Centil_fSUVR_rsf_TOT_CORTMEAN = np.array([i[0] for i in oas_info['Centil_fSUVR_rsf_TOT_CORTMEAN'].squeeze()]).squeeze().astype(float)
    ab_dx = np.array([i[0] for i in oas_info['ab+'].squeeze()])
    ab_dx2 = np.array([i[0] for i in oas_info2['ab+'].squeeze()])
    select_pcd_used_oas = np.c_[[sub, tracer, Centil_fBP_TOT_CORTMEAN, Centil_fSUVR_TOT_CORTMEAN,
                              Centil_fBP_rsf_TOT_CORTMEAN, Centil_fSUVR_rsf_TOT_CORTMEAN, ab_dx]].T

    select_pcd_used_oas[select_pcd_used_oas[:,-1] == 'CN',-1] = '0'
    select_pcd_used_oas[select_pcd_used_oas[:,-1] == 'AD risk',-1] = '1'
    select_pcd_used_oas = np.delete(select_pcd_used_oas, 1, -1).astype(float)
    select_fc_oas = np.expand_dims(select_fc_oas, 0)
    select_fc_all = np.r_[select_fc_oas, aug_fc_oas]
    select_fc_used = select_fc_all.reshape(-1,100,100)
    select_pcd_used = np.tile(select_pcd_used_oas,(opt.augmentation+1,1))

    oas_age = np.array([i[0] for i in oas_info['Age'].squeeze()]).squeeze().astype(float)
    oas_sex = np.array([i[0] for i in oas_info['SEX'].squeeze()]).squeeze()
    oas_sex[oas_sex==2]=0
    oas_age = np.tile(oas_age,(opt.augmentation+1))
    oas_sex = np.tile(oas_sex,(opt.augmentation+1))
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    oas_sex = np.squeeze(imp.fit_transform(np.expand_dims(oas_sex,1)))

    ##################### site correction
    age = np.r_[select_pcd[:,2], oas_age]
    sex = np.r_[select_pcd[:,1], oas_sex]
    symptom = np.r_[select_pcd[:,-1], select_pcd_used[:,4]]
    # label = np.r_[dx_all[~null_mask], label_indep, np.zeros(sum(dx_patient=='SCHZ'))+2, np.zeros(sum(dx_patient=='BIPOLAR'))+3,
    #               np.zeros(sum(dx_patient=='ADHD'))+4]
    site = np.r_[np.ones(select_pcd.shape[0]), np.ones(oas_age.shape[0])+1]
    covars = {'Age': age, 'Site': site, 'symptom': symptom, 'Sex': sex}    
    covars = pd.DataFrame(covars)

    # To specify names of the variables that are categorical:
    categorical_cols = ['Sex']
    # continuous_cols = ['Age', 'symptom']
    continuous_cols = ['Age']
    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = 'Site'

    fc_info = np.r_[select_fc, select_fc_used]
    fc_info = keep_triangle_half(fc_info.shape[1] * (fc_info.shape[1]-1)//2, fc_info.shape[0], fc_info)
    if opt.demean:
        fc_info[-select_fc_used.shape[0]:] = fc_info[-select_fc_used.shape[0]:] - fc_info[-select_fc_used.shape[0]:].mean()
        fc_info[:select_fc.shape[0]] = fc_info[:select_fc.shape[0]] - fc_info[:select_fc.shape[0]].mean()
    # Harmonization step:
    combat = neuroCombat(dat=fc_info.T,
        covars=covars,
        batch_col=batch_col,
        continuous_cols=continuous_cols,
        categorical_cols = categorical_cols)
    data_combat = combat["data"].T
    select_fc_used_half = data_combat[-select_fc_used.shape[0]:]
    select_fc_used, _ = vector_to_matrix(select_fc_used_half)
    
    # ##################### zscore

    scaler = StandardScaler()
    select_fc_used_half = scaler.fit_transform(select_fc_used_half.T).T
    select_fc_used, _ = vector_to_matrix(select_fc_used_half)
    
    
    batch = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []
    pcd = []
    eyes = []
    for j in range(len(select_fc_used)):
        res = get_singal_graph(select_fc_used[j], copy.deepcopy(select_fc_used[j]), 95)
        edge_att_list.append(res[0])
        edge_index_list.append(res[1]+j*res[3])
        att_list.append(res[2])
        batch.append([j]*res[3])
        pseudo.append(np.diag(np.ones(res[3])))
        pcd.append(select_pcd_used[j])
        eyes.append(np.eye(100))
        
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pcd_arr = np.array(pcd)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    eyes_arr = np.concatenate(eyes, axis=0)
    
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    pcd_arr = torch.from_numpy(pcd_arr).float()
    eyes_torch = torch.from_numpy(eyes_arr).float()
    
    data = Data(x=att_torch, edge_index=edge_index_torch, edge_attr=edge_att_torch, pos = pseudo_torch,
                pcd = pcd_arr, eyes = eyes_torch)    
    data, slices = split(data, batch_torch)

    return data, slices


def read_data_regression(opt):
    
    select_fc = sio.loadmat('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_fc.mat')['FC_bl']
    select_pcd_all = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_csfRExtract.csv')
    select_pcd = select_pcd_all[select_pcd_all['TimePoint'] == 'bl']
    # select_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
    ts_file = open(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_ts_bl.pkl', 'rb')
    select_ts = pickle.load(ts_file)
    ts_file.close()
    aug_fc = get_fc_aug(select_ts, 100, opt.augmentation)
    select_pcd_used = select_pcd[['subjectID', 'TAU', 'PTAU', 'ABETA', 'DX']]
    select_pcd_used['subjectID'] = select_pcd_used['subjectID'].str[6:]
    select_fc = select_fc[~select_pcd['ABETA'].isnull()]
    aug_fc = aug_fc[:,~select_pcd['ABETA'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd['ABETA'].isna()]
    select_fc = select_fc[~select_pcd_used['DX'].isnull()]
    aug_fc = aug_fc[:,~select_pcd_used['DX'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd_used['DX'].isnull()]
    
    select_pcd_used = select_pcd_used.values
    # norm_mask = (select_pcd_used[:,-2] >= 977) & (select_pcd_used[:,-3] <= 22)
    norm_mask = (select_pcd_used[:,-2] >= 977)
    select_pcd_used[~norm_mask,-1] = 'AD risk'
    select_pcd_used[norm_mask,-1] = 'CN'
    
    select_fc2 = sio.loadmat(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau3.mat')['fmri_withtau_all_bl']
    select_pcd2 = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
    select_pcd_used2 = select_pcd2[['CONP_CandID', 'tau', 'ptau', 'Amyloid_beta_1_42']]
    ts_file2 = open(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/ts_withtau_bl.pkl', 'rb')
    select_ts2 = pickle.load(ts_file2)
    ts_file2.close()
    aug_fc2 = get_fc_aug(select_ts2, 100, opt.augmentation)
    select_fc2 = select_fc2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())]
    aug_fc2 = aug_fc2[:,~select_pcd_used2['Amyloid_beta_1_42'].isna(),:,:]
    select_pcd_used2 = select_pcd_used2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())].values

    select_dx2 = []
    for i in range(len(select_pcd_used2)):
        # if (select_pcd_used2[i, -1] >= 977 ) and (select_pcd_used2[i, -2] <= 22):
        if (select_pcd_used2[i, -1] >= 977):
            select_dx2.append('CN')
        else:
            select_dx2.append('AD risk')
    select_pcd_used2 = np.c_[select_pcd_used2, select_dx2]

    if opt.demean:
        select_fc = select_fc - select_fc.mean()
        select_fc2 = select_fc2 - select_fc2.mean(-1).mean()

    select_fc_used = np.r_[select_fc, select_fc2]
    select_fc_used = np.expand_dims(select_fc_used, 0)
    select_pcd = np.r_[select_pcd_used, select_pcd_used2]
    select_pcd[select_pcd[:,-1] == 'CN',-1] = '0'
    select_pcd[select_pcd[:,-1] == 'AD risk',-1] = '1'
    select_pcd = np.array(select_pcd).astype(float)
    select_fc_used_aug = np.concatenate((aug_fc, aug_fc2), 1)
    
    # del_pcd = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_csfRExtract_del.csv')
    # del_pcd = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_csfRExtract_del2.csv')
    del_pcd = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_csfRExtract_del_ptau.csv')
    _, _, idx_del = np.intersect1d(del_pcd['subjectID'], select_pcd[:,0], return_indices=True)
    select_fc_used = np.delete(select_fc_used, idx_del, 1)
    select_fc_used_aug = np.delete(select_fc_used_aug, idx_del, 1)
    select_pcd = np.delete(select_pcd, idx_del, 0)
    
    select_fc = np.r_[select_fc_used, select_fc_used_aug]
    select_fc = select_fc.reshape(-1,100,100)
    select_pcd = np.tile(select_pcd,(opt.augmentation+1,1))

    batch = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []
    pcd = []
    eyes = []
    for j in range(len(select_fc)):
        res = get_singal_graph(select_fc[j], copy.deepcopy(select_fc[j]), 95)
        edge_att_list.append(res[0])
        edge_index_list.append(res[1]+j*res[3])
        att_list.append(res[2])
        batch.append([j]*res[3])
        pseudo.append(np.diag(np.ones(res[3])))
        pcd.append(select_pcd[j])
        eyes.append(np.eye(100))
        
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pcd_arr = np.array(pcd)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    eyes_arr = np.concatenate(eyes, axis=0)
    
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    pcd_arr = torch.from_numpy(pcd_arr).float()
    eyes_torch = torch.from_numpy(eyes_arr).float()
    
    data = Data(x=att_torch, edge_index=edge_index_torch, edge_attr=edge_att_torch, pos = pseudo_torch,
                pcd = pcd_arr, eyes = eyes_torch)    
    data, slices = split(data, batch_torch)

    return data, slices


def read_data_regression_pet_ab_adni(opt):
    
    select_fc = sio.loadmat('/home/alex/project/CGCN/A+/dataset/AD/ADNI/adni_raw_pet_fc.mat')['FC_bl']
    select_pcd = pd.read_csv('/home/alex/project/CGCN/A+/dataset/AD/ADNI/adni_raw_pet_pcd_bl.csv')
    del_pcd = pd.read_csv('/home/alex/project/CGCN/A+/dataset/AD/ADNI/adni_raw_pet_pcd_bl_del.csv')
    ts_file = open(r'/home/alex/project/CGCN/A+/dataset/AD/ADNI/adni_raw_pet_ts_bl.pkl', 'rb')
    select_ts = pickle.load(ts_file)
    ts_file.close()
    select_pcd['subjectID'] = select_pcd['subjectID'].str[6:].astype(float)

    sub_used = np.setxor1d(select_pcd['subjectID'], del_pcd['subjectID'])
    _, _, idx_used = np.intersect1d(sub_used, select_pcd['subjectID'], return_indices=True)
    select_pcd = select_pcd.iloc[idx_used]
    select_ts = np.array(select_ts)[idx_used]
    select_fc = select_fc[idx_used]
    
    aug_fc = get_fc_aug(select_ts, 100, opt.augmentation)
    select_pcd_used = select_pcd[['subjectID', 'AV45', 'DX']]
    select_fc = select_fc[~select_pcd['AV45'].isnull()]
    aug_fc = aug_fc[:,~select_pcd['AV45'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd['AV45'].isna()]
    select_pcd_used = select_pcd_used.values
    norm_mask = select_pcd_used[:,-2] < 1.11
    select_pcd_used[~norm_mask,-1] = 'AD risk'
    select_pcd_used[norm_mask,-1] = 'CN'
    select_pcd_used[select_pcd_used[:,-1] == 'CN',-1] = '0'
    select_pcd_used[select_pcd_used[:,-1] == 'AD risk',-1] = '1'
    select_pcd_used = np.array(select_pcd_used).astype(float)
    select_pcd_used = np.tile(select_pcd_used,(opt.augmentation+1,1))
    select_fc_used = np.expand_dims(select_fc, 0)
    select_fc_used = np.r_[select_fc_used, aug_fc]
    select_fc_used = select_fc_used.reshape(-1,100,100)
    select_fc_used_half = keep_triangle_half(select_fc_used.shape[1] * (select_fc_used.shape[1]-1)//2, select_fc_used.shape[0], select_fc_used)
    scaler = StandardScaler()
    select_fc_used_half = scaler.fit_transform(select_fc_used_half.T).T
    select_fc_used, _ = vector_to_matrix(select_fc_used_half)

    batch = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []
    pcd = []
    eyes = []
    for j in range(len(select_fc_used)):
        res = get_singal_graph(select_fc_used[j], copy.deepcopy(select_fc_used[j]), 95)
        edge_att_list.append(res[0])
        edge_index_list.append(res[1]+j*res[3])
        att_list.append(res[2])
        batch.append([j]*res[3])
        pseudo.append(np.diag(np.ones(res[3])))
        pcd.append(select_pcd_used[j])
        eyes.append(np.eye(100))
        
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pcd_arr = np.array(pcd)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    eyes_arr = np.concatenate(eyes, axis=0)
    
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    pcd_arr = torch.from_numpy(pcd_arr).float()
    eyes_torch = torch.from_numpy(eyes_arr).float()
    
    data = Data(x=att_torch, edge_index=edge_index_torch, edge_attr=edge_att_torch, pos = pseudo_torch,
                pcd = pcd_arr, eyes = eyes_torch)    
    data, slices = split(data, batch_torch)

    return data, slices

def read_data_classification(opt):
    
    select_fc = sio.loadmat('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_fc.mat')['FC_bl']
    select_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
    ts_file = open(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_ts_bl.pkl', 'rb')
    select_ts = pickle.load(ts_file)
    ts_file.close()
    aug_fc = get_fc_aug(select_ts, 100, opt.augmentation)
    select_pcd_used = select_pcd[['subjectID', 'TAU', 'PTAU', 'ABETA', 'DX']]
    select_pcd_used['subjectID'] = select_pcd_used['subjectID'].str[6:]
    select_fc = select_fc[~select_pcd['PTAU'].isnull()]
    aug_fc = aug_fc[:,~select_pcd['PTAU'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd['PTAU'].isna()]
    select_fc = select_fc[~select_pcd_used['DX'].isnull()]
    aug_fc = aug_fc[:,~select_pcd_used['DX'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd_used['DX'].isnull()]
    
    select_pcd_used = select_pcd_used.values
    norm_mask = (select_pcd_used[:,-2] >= 977) & (select_pcd_used[:,-3] <= 22)
    dx = np.empty(len(norm_mask),dtype=object)
    dx[~norm_mask] = 'AD risk' 
    dx[norm_mask] = 'CN' 
    select_pcd_used = np.c_[select_pcd_used, dx]
    
    select_fc2 = sio.loadmat(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau3.mat')['fmri_withtau_all_bl']
    select_pcd2 = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
    select_pcd_used2 = select_pcd2[['CONP_CandID', 'tau', 'ptau', 'Amyloid_beta_1_42']]
    ts_file2 = open(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/ts_withtau_bl.pkl', 'rb')
    select_ts2 = pickle.load(ts_file2)
    ts_file2.close()
    aug_fc2 = get_fc_aug(select_ts2, 100, opt.augmentation)
    select_fc2 = select_fc2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())]
    aug_fc2 = aug_fc2[:,~select_pcd_used2['Amyloid_beta_1_42'].isna(),:,:]
    select_pcd_used2 = select_pcd_used2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())].values
    select_dx2 = []
    select_dx_dem = []
    for i in range(len(select_pcd_used2)):
        if (select_pcd_used2[i, -1] >= 977 ) and (select_pcd_used2[i, -2] <= 22):
            select_dx2.append('CN')
        else:
            select_dx2.append('AD risk')
        select_dx_dem.append('CN')
    select_pcd_used2 = np.c_[select_pcd_used2, select_dx_dem, select_dx2]
        
    select_fc_used = np.r_[select_fc, select_fc2]
    select_fc_used = np.expand_dims(select_fc_used, 0)
    select_pcd = np.r_[select_pcd_used, select_pcd_used2]
    select_pcd[select_pcd[:,-1] == 'CN',-1] = '0'
    select_pcd[select_pcd[:,-1] == 'AD risk',-1] = '1'
    select_pcd[select_pcd[:,-2] == 'CN',-2] = '0'
    select_pcd[select_pcd[:,-2] == 'MCI',-2] = '1'
    select_pcd[select_pcd[:,-2] == 'Dementia',-2] = '2'
    select_pcd = np.array(select_pcd).astype(float)
    select_fc_used_aug = np.concatenate((aug_fc, aug_fc2), 1)
    select_fc = np.r_[select_fc_used, select_fc_used_aug]
    select_fc = select_fc.reshape(-1,100,100)
    select_pcd = np.tile(select_pcd,(opt.augmentation+1,1))
    
    if not os.path.exists(opt.save_file):
        os.makedirs(opt.save_file)
    # dignosis_label = select_pcd_bl['DX']

    batch = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []
    pcd = []
    eyes = []
    for j in range(len(select_fc)):
        res = get_singal_graph(select_fc[j], copy.deepcopy(select_fc[j]), 80)
        edge_att_list.append(res[0])
        edge_index_list.append(res[1]+j*res[3])
        att_list.append(res[2])
        batch.append([j]*res[3])
        pseudo.append(np.diag(np.ones(res[3])))
        pcd.append(select_pcd[j])
        eyes.append(np.eye(100))
        
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pcd_arr = np.array(pcd)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    eyes_arr = np.concatenate(eyes, axis=0)
    
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    pcd_arr = torch.from_numpy(pcd_arr).float()
    eyes_torch = torch.from_numpy(eyes_arr).float()
    
    data = Data(x=att_torch, edge_index=edge_index_torch, edge_attr=edge_att_torch, pos = pseudo_torch,
                pcd = pcd_arr, eyes = eyes_torch)    
    data, slices = split(data, batch_torch)

    return data, slices

def read_data_classification_hc_45(opt):
    
    select_fc = sio.loadmat('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_fc.mat')['FC_bl']
    select_pcd_all = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_csfRExtract.csv')
    select_pcd = select_pcd_all[select_pcd_all['TimePoint'] == 'bl']
    ts_file = open(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_ts_bl.pkl', 'rb')
    select_ts = pickle.load(ts_file)
    ts_file.close()
    aug_fc = get_fc_aug(select_ts, 100, opt.augmentation)
    select_pcd_used = select_pcd[['subjectID', 'TAU', 'PTAU', 'ABETA', 'DX']]
    select_pcd_used['subjectID'] = select_pcd_used['subjectID'].str[6:]
    select_fc = select_fc[~select_pcd['PTAU'].isnull()]
    aug_fc = aug_fc[:,~select_pcd['PTAU'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd['PTAU'].isna()]
    select_fc = select_fc[~select_pcd_used['DX'].isnull()]
    aug_fc = aug_fc[:,~select_pcd_used['DX'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd_used['DX'].isnull()]
    
    select_pcd_used = select_pcd_used.values
    norm_mask = (select_pcd_used[:,-2] >= 977) & (select_pcd_used[:,-3] <= 22)
    dx = np.empty(len(norm_mask),dtype=object)
    dx[~norm_mask] = 'AD risk' 
    dx[norm_mask] = 'CN' 
    select_pcd_used = np.c_[select_pcd_used, dx]
    
    sub_without_bl = np.setxor1d(select_pcd_all['subjectID'][~select_pcd_all['ABETA'].isna()].str[6:], select_pcd_used[:,0])
    select_pcd_withptau_more_idx = []
    for sub in sub_without_bl:
        mask_id = np.where(select_pcd_all['subjectID'].str[6:] == sub)[0]
        select_pcd_withptau_more_idx.append(mask_id[~(select_pcd_all['ABETA'].iloc[mask_id].isna())][0])
    select_pcd_withptau_more_idx = np.array(select_pcd_withptau_more_idx)
    select_pcd_more_time = select_pcd_all[['subjectID', 'TAU', 'PTAU', 'ABETA', 'DX']].iloc[select_pcd_withptau_more_idx]
    select_pcd_more_time['subjectID'] = select_pcd_more_time['subjectID'].str[6:]
    select_pcd_more_time = select_pcd_more_time.values
    select_fc_all = sio.loadmat('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_fc.mat')['FC_all']
    ts_file_all = open(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_ts.pkl', 'rb')
    select_ts_all = pickle.load(ts_file_all)
    ts_file_all.close()
    select_fc_more = select_fc_all[select_pcd_withptau_more_idx]
    ts_more = np.array(select_ts_all)[select_pcd_withptau_more_idx]
    aug_fc_more = get_fc_aug(ts_more, 100, opt.augmentation)
    norm_mask_more = (select_pcd_more_time[:,-2] >= 977) & (select_pcd_more_time[:,-3] <= 22)
    dx = np.empty(len(norm_mask_more),dtype=object)
    dx[~norm_mask_more] = 'AD risk' 
    dx[norm_mask_more] = 'CN' 
    select_pcd_more_time = np.c_[select_pcd_more_time, dx]
    
    select_pcd_used = np.r_[select_pcd_used, select_pcd_more_time]
    select_fc = np.r_[select_fc, select_fc_more]
    aug_fc = np.concatenate((aug_fc, aug_fc_more), 1)
    
    select_fc2 = sio.loadmat(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau3.mat')['fmri_withtau_all_bl']
    select_pcd2 = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
    select_pcd_used2 = select_pcd2[['CONP_CandID', 'tau', 'ptau', 'Amyloid_beta_1_42']]
    ts_file2 = open(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/ts_withtau_bl.pkl', 'rb')
    select_ts2 = pickle.load(ts_file2)
    ts_file2.close()
    aug_fc2 = get_fc_aug(select_ts2, 100, opt.augmentation)
    select_fc2 = select_fc2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())]
    aug_fc2 = aug_fc2[:,~select_pcd_used2['Amyloid_beta_1_42'].isna(),:,:]
    select_pcd_used2 = select_pcd_used2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())].values
    select_dx2 = []
    select_dx_dem = []
    for i in range(len(select_pcd_used2)):
        if (select_pcd_used2[i, -1] >= 977 ) and (select_pcd_used2[i, -2] <= 22):
            select_dx2.append('CN')
        else:
            select_dx2.append('AD risk')
        select_dx_dem.append('CN')
    select_pcd_used2 = np.c_[select_pcd_used2, select_dx_dem, select_dx2]

    # if opt.demean:
    #     select_fc = select_fc - select_fc.mean()
    #     select_fc2 = select_fc2 - select_fc2.mean(-1).mean()
    #     aug_fc = aug_fc - aug_fc.mean()
    #     aug_fc2 = aug_fc2 - aug_fc2.mean()

        
    select_fc_used = np.r_[select_fc, select_fc2]
    select_fc_used = np.expand_dims(select_fc_used, 0)
    select_pcd = np.r_[select_pcd_used, select_pcd_used2]
    select_pcd[select_pcd[:,-1] == 'CN',-1] = '0'
    select_pcd[select_pcd[:,-1] == 'AD risk',-1] = '1'
    select_pcd[select_pcd[:,-2] == 'CN',-2] = '0'
    select_pcd[select_pcd[:,-2] == 'MCI',-2] = '1'
    select_pcd[select_pcd[:,-2] == 'Dementia',-2] = '2'
    select_pcd = np.array(select_pcd).astype(float)
    select_fc_used_aug = np.concatenate((aug_fc, aug_fc2), 1)
    select_fc = np.r_[select_fc_used, select_fc_used_aug]
    select_fc = select_fc.reshape(-1,100,100)
    select_pcd = np.tile(select_pcd,(opt.augmentation+1,1))
    
    # select_fc_used_half = keep_triangle_half(select_fc.shape[1] * (select_fc.shape[1]-1)//2, select_fc.shape[0], select_fc)
    # scaler = StandardScaler()
    # select_fc_used_half = scaler.fit_transform(select_fc_used_half.T).T
    # select_fc, _ = vector_to_matrix(select_fc_used_half)
    
    if not os.path.exists(opt.save_file):
        os.makedirs(opt.save_file)
    # dignosis_label = select_pcd_bl['DX']

    batch = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []
    pcd = []
    eyes = []
    for j in range(len(select_fc)):
        res = get_singal_graph(select_fc[j], copy.deepcopy(select_fc[j]), 80)
        edge_att_list.append(res[0])
        edge_index_list.append(res[1]+j*res[3])
        att_list.append(res[2])
        batch.append([j]*res[3])
        pseudo.append(np.diag(np.ones(res[3])))
        pcd.append(select_pcd[j])
        eyes.append(np.eye(100))
        
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pcd_arr = np.array(pcd)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    eyes_arr = np.concatenate(eyes, axis=0)
    
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    pcd_arr = torch.from_numpy(pcd_arr).float()
    eyes_torch = torch.from_numpy(eyes_arr).float()
    
    data = Data(x=att_torch, edge_index=edge_index_torch, edge_attr=edge_att_torch, pos = pseudo_torch,
                pcd = pcd_arr, eyes = eyes_torch)    
    data, slices = split(data, batch_torch)

    return data, slices

def read_data_classification_convert(opt):
    
    pcd_all = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd.csv')
    # pcd_all = pcd_all[~pcd_all['DX'].isnull()]
    pcd_all = pcd_all[~pcd_all['MOCA'].isnull()]
    pcd_all['DX'][pcd_all['DX'] == 'AD'] = 'Dementia'
    pcd_name, pcd_ida, pcd_idb = np.unique(pcd_all['subjectID'], return_index = True, return_inverse = True)
    convert_label = []
    for i in range(len(pcd_name)):
        pcd_dx = pcd_all['MOCA'][pcd_idb == i]
        # pcd_dx = adni_cdr[pcd_idb == i]
        # pcd_dx_diff = pcd_dx - pcd_dx[0]
        pcd_dx_diff = pcd_dx - pcd_dx.iloc[0]
        if pcd_dx_diff.sum()>=0:
            convert_label.append(pcd_dx_diff.max())
        else:
            convert_label.append(pcd_dx_diff.min())

    convert_label = np.array(convert_label)            
    select_fc = sio.loadmat('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_fc.mat')['FC_bl']
    select_pcd = pd.read_csv('/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_pcd_bl.csv')
    select_name = select_pcd['subjectID']
    s_name, idx_bl, idx_all = np.intersect1d(select_name, pcd_name, return_indices=True)
    from collections import Counter
    print(Counter(convert_label[idx_all]))
    
    ts_file = open(r'/home/alex/project/CGCN/dataset/AD/ADNI/adni_all_ts_bl.pkl', 'rb')
    select_ts = pickle.load(ts_file)
    ts_file.close()
    aug_fc = get_fc_aug(select_ts, 100, opt.augmentation)
    select_pcd_used = select_pcd[['subjectID', 'TAU', 'PTAU', 'ABETA', 'DX']]
    select_pcd_used['subjectID'] = select_pcd_used['subjectID'].str[6:]
    select_fc = select_fc[~select_pcd['PTAU'].isnull()]
    aug_fc = aug_fc[:,~select_pcd['PTAU'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd['PTAU'].isna()]
    select_fc = select_fc[~select_pcd_used['DX'].isnull()]
    aug_fc = aug_fc[:,~select_pcd_used['DX'].isnull(),:,:]
    select_pcd_used = select_pcd_used[~select_pcd_used['DX'].isnull()]
    
    select_pcd_used = select_pcd_used.values
    norm_mask = (select_pcd_used[:,-2] >= 977) & (select_pcd_used[:,-3] <= 22)
    dx = np.empty(len(norm_mask),dtype=object)
    dx[~norm_mask] = 'AD risk' 
    dx[norm_mask] = 'CN' 
    select_pcd_used = np.c_[select_pcd_used, dx]
    
    select_fc2 = sio.loadmat(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau3.mat')['fmri_withtau_all_bl']
    select_pcd2 = pd.read_csv(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/fmri_withtau_pcd_bl3.csv')
    select_pcd_used2 = select_pcd2[['CONP_CandID', 'tau', 'ptau', 'Amyloid_beta_1_42']]
    ts_file2 = open(r'/home/alex/project/CGCN/dataset/AD/PREVENT-AD/ts_withtau_bl.pkl', 'rb')
    select_ts2 = pickle.load(ts_file2)
    ts_file2.close()
    aug_fc2 = get_fc_aug(select_ts2, 100, opt.augmentation)
    select_fc2 = select_fc2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())]
    aug_fc2 = aug_fc2[:,~select_pcd_used2['Amyloid_beta_1_42'].isna(),:,:]
    select_pcd_used2 = select_pcd_used2[~(select_pcd_used2['Amyloid_beta_1_42'].isna())].values
    select_dx2 = []
    select_dx_dem = []
    for i in range(len(select_pcd_used2)):
        if (select_pcd_used2[i, -1] >= 977 ) and (select_pcd_used2[i, -2] <= 22):
            select_dx2.append('CN')
        else:
            select_dx2.append('AD risk')
        select_dx_dem.append('CN')
    select_pcd_used2 = np.c_[select_pcd_used2, select_dx_dem, select_dx2]

    select_fc_used = np.r_[select_fc, select_fc2]
    select_fc_used = np.expand_dims(select_fc_used, 0)
    select_pcd = np.r_[select_pcd_used, select_pcd_used2]
    select_pcd[select_pcd[:,-1] == 'CN',-1] = '0'
    select_pcd[select_pcd[:,-1] == 'AD risk',-1] = '1'
    select_pcd[select_pcd[:,-2] == 'CN',-2] = '0'
    select_pcd[select_pcd[:,-2] == 'MCI',-2] = '1'
    select_pcd[select_pcd[:,-2] == 'Dementia',-2] = '2'
    select_pcd = np.array(select_pcd).astype(float)
    select_fc_used_aug = np.concatenate((aug_fc, aug_fc2), 1)
    select_fc = np.r_[select_fc_used, select_fc_used_aug]
    select_fc = select_fc.reshape(-1,100,100)
    select_pcd = np.tile(select_pcd,(opt.augmentation+1,1))
    
    if not os.path.exists(opt.save_file):
        os.makedirs(opt.save_file)
    # dignosis_label = select_pcd_bl['DX']

    batch = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []
    pcd = []
    eyes = []
    for j in range(len(select_fc)):
        res = get_singal_graph(select_fc[j], copy.deepcopy(select_fc[j]), 80)
        edge_att_list.append(res[0])
        edge_index_list.append(res[1]+j*res[3])
        att_list.append(res[2])
        batch.append([j]*res[3])
        pseudo.append(np.diag(np.ones(res[3])))
        pcd.append(select_pcd[j])
        eyes.append(np.eye(100))
        
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pcd_arr = np.array(pcd)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    eyes_arr = np.concatenate(eyes, axis=0)
    
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    pcd_arr = torch.from_numpy(pcd_arr).float()
    eyes_torch = torch.from_numpy(eyes_arr).float()
    
    data = Data(x=att_torch, edge_index=edge_index_torch, edge_attr=edge_att_torch, pos = pseudo_torch,
                pcd = pcd_arr, eyes = eyes_torch)    
    data, slices = split(data, batch_torch)

    return data, slices
