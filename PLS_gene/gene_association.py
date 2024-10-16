#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:52:41 2023

@author: alex
"""
import abagen
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from statsmodels.stats.multitest import multipletests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import cm
import matplotlib as mpl
from scipy.stats import pearsonr
import multiprocessing
from functools import partial
from collections import Counter
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.weight']= 'bold'
matplotlib.rcParams['font.size']= 30

class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

###################################
#######################################
###################################csf ab dis specific
# expression, report = abagen.get_expression_data('/home/alex/project/utils_for_all/parcellation/SchaeferYeo2018/MNI/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', 
#                                                 return_report=True, sample_norm = 'zscore', gene_norm = 'zscore')
expression = pd.read_csv(r'/home/alex/project/CGCN/A+/final_figure/gene_analysis/Gene_expression_zscore.csv')
expression = expression.drop(columns = ['Unnamed: 0'])
importance = sio.loadmat(r'/home/alex/project/CGCN/A+/final_figure/ab/salient_node_importance_dis_raw_notrans_A+')['node_strength'].squeeze()
# gradient_info = sio.loadmat(r'/home/alex/project/CGCN/A+/final_figure/ab/raw/Saliency_node_edge_strength_ab_raw')
# importance = gradient_info['node_importance_raw_reshape_dis_all'].sum(0).sum(0).sum(0)
# importance = (importance - importance.mean())/importance.std()
pls = PLSRegression(n_components=1)
pls.fit(expression, importance)
weight_raw = pls.coef_.squeeze()
r2_raw = pls.score(expression, importance)
explain_raw = sum(abs(pls.x_scores_)*abs(pls.x_scores_),1) / sum(sum(abs(expression.values)*abs(expression.values),1))
########boostrap
weights_all = []
for i in range(2000):
    print(i)
    idx = np.random.randint(0, len(importance), size=(len(importance), 1)).squeeze()
    expression_boost = expression.iloc[idx]
    importance_boost = importance[idx]
    pls = PLSRegression(n_components=1)
    pls.fit(expression_boost, importance_boost)
    weight = pls.coef_
    weights_all.append(weight)
weights_all = np.array(weights_all).squeeze()
p_mask = weights_all > 0
p_number = np.sum(p_mask, 0)
p_boost = p_number / 2000
p_boost[p_boost>0.5] = 1-p_boost[p_boost>0.5]
p_boost = p_boost*2
p_fdr = multipletests(p_boost, method="fdr_bh")[1]
zscore = weight_raw/np.std(weights_all,0)
table = {'Gene': expression.keys(), 'Z-score': zscore, 'P-fdr': p_fdr}
df = pd.DataFrame(table)
df_significant = df[df['P-fdr']<=0.05]
idx_min_vis = (zscore.argsort()[:5])
idx_max_vis = (zscore.argsort()[-5:])
df_vis = df.iloc[np.r_[idx_min_vis, idx_max_vis]]
COL = MplColorHelper('RdBu_r', df_vis['Z-score'].min(), df_vis['Z-score'].max())
c = COL.get_rgb(np.arange(-4,6,1))
plt.figure(figsize=(10,10)) 
ax = plt.gca()#获取边框
for i, val in enumerate(df_vis['Z-score']):
    plt.text(0, i/3, df_vis['Gene'].iloc[i], va='center', ha='center', fontsize=100, c = c[i])
    plt.text(1, i/3, "%.3f"%val, va='center', ha='center', fontsize=100, c = c[i])
plt.text(0, i/3+0.4, 'Gene', va='center', ha='center', fontsize=100)
plt.text(1, i/3+0.4, 'Z-score', va='center', ha='center', fontsize=100)
plt.axis('off')
# plt.savefig(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/Gene_zscore_salient_node_importance_dis_raw_notrans_A+.svg', format = 'svg', bbox_inches = 'tight')
expression_top_visual = {gene: expression[gene].values for gene in df_vis['Gene']}
# expression.to_csv(r'/home/alex/project/CGCN/A+/final_figure/ab/Gene_expression.csv')
df_vis.to_csv(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/Gene_zscore_salient_node_importance_dis_raw_notrans_A+_top10.csv')
df.to_csv(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/Gene_zscore_salient_node_importance_dis_raw_notrans_A+.csv')
df_significant.to_csv(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/Gene_zscore_salient_node_importance_dis_raw_notrans_A+_significant.csv', index=False)
sio.savemat(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/Gene_zscore_salient_node_importance_dis_raw_notrans_A+_top10.mat', expression_top_visual)
#########permutation
r2_all = []
var_all = []
weights_all_permute = []
for j in range(1000):
    print(j)
    idx = np.arange(100)
    np.random.shuffle(idx)
    importance_shuffle = importance[idx]
    pls = PLSRegression(n_components=1)
    pls.fit(expression, importance_shuffle)
    weight = pls.coef_
    weights_all_permute.append(weight)
    r2 = pls.score(expression, importance_shuffle)
    explain = sum(abs(pls.x_scores_)*abs(pls.x_scores_),1) / sum(sum(abs(expression.values)*abs(expression.values),1))
    r2_all.append(r2)
    var_all.append(explain)
r2_all = np.array(r2_all).squeeze()
var_all = np.array(var_all).squeeze()
weights_all_permute = np.array(weights_all_permute).squeeze()
# sio.savemat(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/dis_specific/PLS_permute.mat', {'var_permute_all': var_all, 'var_true': explain_raw,
#                                                                                               'r2_all': r2_all, 'r2_true': r2_raw})

#########permutation cell counts
def boostrap(seed):
    idx = np.arange(100)
    np.random.shuffle(idx)
    np.random.seed(seed)
    importance_shuffle = importance[idx]
    weights_all = []
    for i in range(1000):
        print(i)
        idx = np.random.randint(0, len(importance), size=(len(importance), 1)).squeeze()
        expression_boost = expression.iloc[idx]
        importance_boost = importance_shuffle[idx]
        pls = PLSRegression(n_components=1)
        pls.fit(expression_boost, importance_boost)
        weight = pls.coef_
        weights_all.append(weight)
    weights_all = np.array(weights_all).squeeze()
    p_mask = weights_all > 0
    p_number = np.sum(p_mask, 0)
    p_boost = p_number / 1000
    p_boost[p_boost>0.5] = 1-p_boost[p_boost>0.5]
    p_boost = p_boost*2
    p_fdr = multipletests(p_boost, method="fdr_bh")[1]
    return p_fdr

p_all = []
for j in range(100):
    print(j)
    idx = np.arange(100)
    np.random.shuffle(idx)
    importance_shuffle = importance[idx]
    weights_all = []
    for i in range(500):
        # print(i)
        idx = np.random.randint(0, len(importance), size=(len(importance), 1)).squeeze()
        expression_boost = expression.iloc[idx]
        importance_boost = importance_shuffle[idx]
        pls = PLSRegression(n_components=1)
        pls.fit(expression_boost, importance_boost)
        weight = pls.coef_
        weights_all.append(weight)
    weights_all = np.array(weights_all).squeeze()
    p_mask = weights_all > 0
    p_number = np.sum(p_mask, 0)
    p_boost = p_number / 500
    p_boost[p_boost>0.5] = 1-p_boost[p_boost>0.5]
    p_boost = p_boost*2
    p_fdr = multipletests(p_boost, method="fdr_bh")[1]
    p_all.append(p_fdr)

p_all = np.array(p_all)
gene_name = np.array(expression.keys())
gene_name = np.tile(gene_name, [100, 1])
gene_name_significant = [] 
for k in range(p_all.shape[0]):
    gene_name_significant.append(gene_name[k,p_all[k]<=0.05])

cell_info = pd.read_csv(r'/home/alex/project/CGCN/A+/final_figure/gene_analysis/celltypes_PSP.csv')
cell_type_gene = cell_info['gene']
cell_type = cell_info['class']
cell_name = cell_type.unique()

gene_true = pd.read_csv(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/dis_specific/Gene_zscore_salient_node_importance_dis_raw_notrans_A+_significant.csv')['Gene']
cell_ture_type = []
cell_ture_name = []
for g in gene_true:
    if (cell_type_gene == g).sum() !=0:
        cell_ture_type.append(cell_type[cell_type_gene == g].iloc[0])
        cell_ture_name.append(cell_type_gene[cell_type_gene == g].iloc[0])
cell_ture_type = np.array(cell_ture_type)
cell_ture_name = np.array(cell_ture_name)

cell_type_gene_expression = pd.concat([pd.DataFrame({'{}'.format(cell_name[0]): cell_ture_name[cell_ture_type == cell_name[0]]}),
                            pd.DataFrame({'{}'.format(cell_name[1]): cell_ture_name[cell_ture_type == cell_name[1]]}),
                            pd.DataFrame({'{}'.format(cell_name[2]): cell_ture_name[cell_ture_type == cell_name[2]]}),
                            pd.DataFrame({'{}'.format(cell_name[3]): cell_ture_name[cell_ture_type == cell_name[3]]}),
                            pd.DataFrame({'{}'.format(cell_name[4]): cell_ture_name[cell_ture_type == cell_name[4]]}),
                            pd.DataFrame({'{}'.format(cell_name[5]): cell_ture_name[cell_ture_type == cell_name[5]]}),
                            pd.DataFrame({'{}'.format(cell_name[6]): cell_ture_name[cell_ture_type == cell_name[6]]})], axis = 1)
# cell_type_gene_expression.to_csv(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/dis_specific/cell_type/cell_type_gene_expression_overlapping.csv', index = False)
count_Astro = np.sum(cell_type == cell_name[0])
count_Endo = np.sum(cell_type == cell_name[1])
count_Micro = np.sum(cell_type == cell_name[2])
count_NeuroEx = np.sum(cell_type == cell_name[3])
count_NeuroIn = np.sum(cell_type == cell_name[4])
count_Oligo = np.sum(cell_type == cell_name[5])
count_OPC = np.sum(cell_type == cell_name[6])

cell_ture_type_info = Counter(cell_ture_type)
cell_ture_type_info_count = np.array(list(cell_ture_type_info.values()))
cell_permute_type_all = []
for gene_one_permute in gene_name_significant:
    cell_permute_type = []
    for g in gene_one_permute:
        if (cell_type_gene == g).sum() !=0:
            cell_permute_type.append(cell_type[cell_type_gene == g].iloc[0])
    cell_permute_type = np.array(cell_permute_type)
    cell_permute_type_info = Counter(cell_permute_type)
    cell_permute_type_info_ = np.zeros((7,1))
    cell_permute_type_info_[0] = cell_permute_type_info['Astro']
    cell_permute_type_info_[1] = cell_permute_type_info['Endo']
    cell_permute_type_info_[2] = cell_permute_type_info['Micro']
    cell_permute_type_info_[3] = cell_permute_type_info['Neuro-Ex']
    cell_permute_type_info_[4] = cell_permute_type_info['Neuro-In']
    cell_permute_type_info_[5] = cell_permute_type_info['Oligo']
    cell_permute_type_info_[6] = cell_permute_type_info['OPC']
    cell_permute_type_all.append(cell_permute_type_info_)
cell_permute_type_all = np.array(cell_permute_type_all).squeeze()
cell_type_info = {'cell_permute_result': cell_permute_type_all, 'cell_name': cell_name, 'cell_ture_result': cell_ture_type_info_count}
# sio.savemat(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/dis_specific/cell_type/cell_type_info.mat', cell_type_info)
p_Astro = (sum(cell_ture_type_info_count[0] < cell_permute_type_all[:,0]) + 1) / 101 
p_Endo = (sum(cell_ture_type_info_count[1] < cell_permute_type_all[:,1]) + 1)/101
p_Micro = (sum(cell_ture_type_info_count[2] < cell_permute_type_all[:,2]) + 1)/101
p_Neuro_Ex = (sum(cell_ture_type_info_count[3] < cell_permute_type_all[:,3]) + 1)/101
p_Neuro_in = (sum(cell_ture_type_info_count[4] < cell_permute_type_all[:,4]) + 1)/101
p_Oligo = (sum(cell_ture_type_info_count[5] < cell_permute_type_all[:,5]) + 1)/101
p_OPC = (sum(cell_ture_type_info_count[6] < cell_permute_type_all[:,6]) + 1)/101

names = ['Astrocytes', 'Excitatory neurons', 'Oligodendrocytes', 'Endothelial', 'Inhibitory neurons', 'Microglia', 'OPCs']
df = pd.DataFrame({'names': names, 'val': cell_ture_type_info_count})
df = pd.DataFrame({'names': names, 'val': 100*cell_ture_type_info_count/np.array([count_Astro, count_NeuroEx, count_Oligo, count_Endo, count_NeuroIn, count_Micro, count_OPC])})
result = df.groupby(["names"])['val'].aggregate(np.median).reset_index().sort_values('val', ascending=False)

plt.figure(figsize=(15,15)) 
ax = plt.gca()#获取边框
sns.barplot(data=df, y="val", x="names", palette = 'Set3', ax=ax, order=result['names'])
plt.ylim(0,20)
loca_x = 5
x_major_locator=MultipleLocator(loca_x)
ax.yaxis.set_major_locator(x_major_locator)
ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
change_width(ax, .55)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")
plt.legend('',frameon=False)
plt.savefig(r'/home/alex/project/CGCN/A+/final_figure/ab/gene/dis_specific/cell_type/cell_type_significant_express_percent.svg', format = 'svg', bbox_inches = 'tight')
