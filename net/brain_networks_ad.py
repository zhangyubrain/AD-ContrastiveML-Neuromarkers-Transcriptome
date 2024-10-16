import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,NNConv,EdgeConv,MessagePassing,DNAConv, GINEConv, GCNConv, GINConv, GENConv
from torch_geometric.nn.conv import GCNConv

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as gsp
from torch_geometric.utils import (add_self_loops, sort_edge_index,dropout_adj, degree, remove_self_loops, to_dense_adj, dense_to_sparse)
from torch_sparse import spspmm

from torch_geometric.nn import SAGPooling, TopKPooling, ASAPooling, dense_mincut_pool
from net.knnGraph_dyconv import DilatedKnnGraph, ResDynBlock, EdgConv
from torch.nn.modules.module import Module
from torch_geometric.utils.repeat import repeat
from numpy.linalg import eig, inv
import numpy as np
from typing import List, Callable, Union, Any, TypeVar, Tuple
from net.mlp import MLP
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse

Tensor = TypeVar('torch.tensor')

###models###

class ContrativeNet(torch.nn.Module):
    def __init__(self, opt):
        super(ContrativeNet, self).__init__()
     
        # self.HC_net = GAE1(opt, True)
        # self.disorder_net = GAE1(opt, True)
        self.HC_net = GAE2(opt, True)
        self.disorder_net = GAE2(opt, True)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.bias is not None:
                module.bias.data.zero_()
            
    # def forward(self, x1, edge_index1, edge_attr1, pcd1, x2, edge_index2, edge_attr2, pcd2, eyes, batch, pretrain):
    def forward(self, x1, edge_index1, edge_attr1, pcd1, x2, edge_index2, edge_attr2, pcd2, eyes, batch):
        '''
        

        Parameters
        ----------
        x1 : disorder
            DESCRIPTION.
        x2 : HC
            DESCRIPTION.

        Returns
        -------
        mu_hc : TYPE
            DESCRIPTION.
        log_var_hc : TYPE
            DESCRIPTION.
        z_hc : TYPE
            DESCRIPTION.
        out_hc : TYPE
            DESCRIPTION.
        mu_dis : TYPE
            DESCRIPTION.
        log_var_dis : TYPE
            DESCRIPTION.
        z_dis : TYPE
            DESCRIPTION.
        out_dis : TYPE
            DESCRIPTION.
        mu_dis_hc : TYPE
            DESCRIPTION.
        log_var_dis_hc : TYPE
            DESCRIPTION.
        z_dis_hc : TYPE
            DESCRIPTION.
        out_dis_hc : TYPE
            DESCRIPTION.

        '''
        # if pretrain:
        output_dis, h_dis, y_dis, mu_dis, log_var_dis = self.disorder_net(x2, edge_index2, edge_attr2, pcd2, eyes, batch)
        output_hc_share, h_hc_share, y_hc_share, mu_hc_share, log_hc_share = self.disorder_net(x1, edge_index1, edge_attr1, pcd1, eyes, batch)
        output_dis_share, h_dis_share, y_pred_dis_share, mu_dis_share, log_var_dis_share = self.HC_net(x2, edge_index2,
                                                                                      edge_attr2, pcd2, eyes, batch)
        output_hc, h_hc, y_pred_hc, mu_hc, log_var_hc = self.HC_net(x1, edge_index1, edge_attr1, pcd1, eyes, batch)
        
        return output_dis, h_dis, y_dis, mu_dis, log_var_dis, output_dis_share, h_dis_share, y_pred_dis_share, \
            mu_dis_share, log_var_dis_share, output_hc, h_hc, y_pred_hc, mu_hc, log_var_hc, output_hc_share, h_hc_share, \
                y_hc_share, mu_hc_share, log_hc_share
        # else:
        # output_dis, h_dis, y_dis, mu_dis, log_var_dis = self.disorder_net(x2, edge_index2, edge_attr2, pcd2, eyes, batch)
        # output_dis_share, h_dis_share, y_pred_dis_share, mu_dis_share, log_var_dis_share = self.HC_net(x2, edge_index2,
        #                                                                              edge_attr2, pcd2, eyes, batch)
        # output_hc, h_hc, y_pred_hc, mu_hc, log_var_hc = self.HC_net(x1, edge_index1, edge_attr1, pcd1, eyes, batch)
        
        # return output_dis, h_dis, y_dis, mu_dis, log_var_dis, output_dis_share, h_dis_share, y_pred_dis_share, \
        #     mu_dis_share, log_var_dis_share, output_hc, h_hc, y_pred_hc, mu_hc, log_var_hc  

class GAE1(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, opt, more_layer):
        super().__init__()
        self.in_channels = opt.in_channels
        self.hidden_channels = opt.hidden_channels
        self.sum_res = opt.sum_res
        self.task = opt.task
        
        if more_layer:
            indim = opt.hidden_channels//2
        else:
            indim = opt.hidden_channels
        opt.indim = indim
        self.encoder = Encoder1(opt, more_layer)
        self.decoder = Decoder1(opt, indim, opt.in_channels, more_layer)

        # self.bn = torch.nn.BatchNorm1d(opt.in_channels)
        if self.task == 'classification':
            self.target_predict = nn.Linear(opt.in_channels, 2)
        elif self.task == 'regression':
            self.target_predict = nn.Linear(opt.in_channels, 1)
        self.fc_mu = nn.Linear(opt.in_channels, indim)
        self.fc_var = nn.Linear(opt.in_channels, indim)
        self.disentangle = nn.Linear(opt.in_channels, 2)
        # self.target_predict2 = nn.Linear(indim, 10)
        # self.target_predict3 = nn.Linear(10, 1)
        self.z_decode = nn.Linear(indim, opt.in_channels)
        if opt.conv == 'gat':
            self.conv_read_reverse = GATConv(1, indim, improved=True, concat=False, dropout=opt.drop, heads=4)
        elif opt.conv == 'edgeconv':
            self.conv_read_reverse = EdgConv(1, indim, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gcn':
            self.conv_read_reverse = GCNConv(1, indim, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gen':
            self.conv_read_reverse = GENConv(1, indim, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gin':
            self.conv_read_reverse = GINConv(MLP([1, indim], act = 'tanh', drop=opt.drop, bias= True))
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x, edge_index, edge_attr, pcd, eyes, batch=None):
                
        h, batch = self.encoder(x, edge_index, edge_attr, eyes, batch)
        h_disentangle = self.disentangle(h)
        h_disentangle = F.sigmoid(h_disentangle)

        y_pred = self.target_predict(h)
        if self.task == 'classification':
            y_pred = F.softmax(y_pred,-1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        z1 = self.reparameterize(mu, log_var)

        z = self.z_decode(z1)
        z = z.view(z.shape[0]*z.shape[1], -1)
        x = self.conv_read_reverse(z, edge_index)

        x = self.decoder(x, edge_index, eyes, batch)
        return x, h_disentangle, y_pred, mu, log_var

class Encoder1(Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, opt, more_layer):
        super(Encoder1, self).__init__()
        self.depth = opt.depth
        if opt.act:
            self.act = F.leaky_relu
        self.down_convs = torch.nn.ModuleList()
        if opt.conv == 'gat':
            self.down_convs.append(GATConv(opt.in_channels, opt.hidden_channels, improved=True, concat=False, dropout=opt.drop, heads=4))
            if more_layer:
                self.down_convs.append(GATConv(opt.hidden_channels, opt.hidden_channels//2, improved=True, concat=False, dropout=opt.drop, heads=4))
        elif opt.conv == 'edgeconv':
            self.down_convs.append(EdgConv(opt.in_channels, opt.hidden_channels, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.down_convs.append(EdgConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
                # self.down_convs.append(EdgConv(opt.hidden_channels//2, opt.hidden_channels//4, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gen':
            self.down_convs.append(GENConv(opt.in_channels, opt.hidden_channels, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.down_convs.append(GENConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gcn':
            self.down_convs.append(GCNConv(opt.in_channels, opt.hidden_channels, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.down_convs.append(GCNConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gin':
            self.down_convs.append(GINConv(MLP([opt.in_channels, opt.hidden_channels], act = 'tanh', drop=opt.drop, bias= True)))
            if more_layer:
                self.down_convs.append(GINConv(MLP([opt.hidden_channels, opt.hidden_channels//2], act = 'tanh', drop=opt.drop, bias= True)))
        if opt.conv == 'gat':
            self.conv_read = GATConv(opt.indim, 1, improved=True, concat=False, dropout=opt.drop, heads=4)
        elif opt.conv == 'edgeconv':
            self.conv_read = EdgConv(opt.indim, 1, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gcn':
            self.conv_read = GCNConv(opt.indim, 1, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gen':
            self.conv_read = GENConv(opt.indim, 1, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gin':
            self.conv_read = GINConv(MLP([opt.indim, 1], act = 'tanh', drop=opt.drop, bias= True))
            
    def forward(self, x, edge_index, edge_attr, eyes, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # adj_get = to_dense_adj(edge_index, batch)#raw edge matrix

        for i in range(len(self.down_convs)):
            x = self.down_convs[i](x, edge_index)
            x = self.act(x)
        
        h = self.conv_read(x, edge_index)
        # h = F.sigmoid(h)
        h = h.view(-1,eyes.shape[-1])

        # h = F.dropout(h, p=0.3, training=self.training)
        return h, batch

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

            
class Decoder1(Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, opt, hidden_channels, out_channels, more_layer):
        super(Decoder1, self).__init__()
        self.sum_res = opt.sum_res
        if opt.act:
            self.act = F.leaky_relu
        self.depth = opt.depth
        self.up_convs = torch.nn.ModuleList()
        if opt.conv == 'gat':
            self.up_convs.append(GATConv(hidden_channels, hidden_channels*2, improved=True, concat=False, dropout=opt.drop, heads=4))
            if more_layer:
                self.up_convs.append(GATConv(hidden_channels*2, out_channels, improved=True, concat=False, dropout=opt.drop, heads=4))
        elif opt.conv == 'edgeconv':
            self.up_convs.append(EdgConv(hidden_channels, hidden_channels*2, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.up_convs.append(EdgConv(hidden_channels*2, out_channels, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gcn':
            self.up_convs.append(GCNConv(hidden_channels, hidden_channels*2, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.up_convs.append(GCNConv(hidden_channels*2, out_channels, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gen':
            self.up_convs.append(GENConv(hidden_channels, hidden_channels*2, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.up_convs.append(GENConv(hidden_channels*2, out_channels, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gin':
            self.up_convs.append(GINConv(MLP([hidden_channels, hidden_channels*2], act = 'tanh', drop=opt.drop, bias= True)))
            if more_layer:
                self.up_convs.append(GINConv(MLP([hidden_channels*2, out_channels], act = 'tanh', drop=opt.drop, bias= True)))

        self.act_final = F.tanh

    def forward(self, x, edge_index, eyes, batch = None):
        
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x, edge_index)
        # x = self.up_convs[1](x, edge_index)
        x_ = x.view(x.shape[0]//100,100,-1)
        x_t = x_.permute(0,2,1)
        x2 = torch.bmm(x_, x_t) 
        x = x2.view(-1,100)
        diag_e = x * eyes
        x = x - diag_e

        return x
    
class GAE2(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, opt, more_layer):
        super().__init__()
        self.in_channels = opt.in_channels
        self.hidden_channels = opt.hidden_channels
        self.sum_res = opt.sum_res
        self.task = opt.task
        
        if more_layer:
            indim = opt.hidden_channels//2
        else:
            indim = opt.hidden_channels
        opt.indim = indim
        self.encoder = Encoder2(opt, more_layer)
        self.decoder = Decoder2(opt, indim, opt.in_channels, more_layer)

        # self.bn = torch.nn.BatchNorm1d(opt.in_channels)
        if self.task == 'classfication':
            self.target_predict = nn.Linear(opt.in_channels, 2)
        elif self.task == 'regression':
            self.target_predict = nn.Linear(opt.in_channels, 1)
        self.fc_mu = nn.Linear(opt.in_channels, indim)
        self.fc_var = nn.Linear(opt.in_channels, indim)
        self.disentangle = nn.Linear(opt.in_channels, 2)
        # self.target_predict2 = nn.Linear(indim, 10)
        # self.target_predict3 = nn.Linear(10, 1)
        self.z_decode = nn.Linear(indim, opt.in_channels)
        if opt.conv == 'gat':
            self.conv_read_reverse = GATConv(1, indim, improved=True, concat=False, dropout=opt.drop, heads=4)
        elif opt.conv == 'edgeconv':
            self.conv_read_reverse = EdgConv(1, indim, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gcn':
            self.conv_read_reverse = GCNConv(1, indim, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gen':
            self.conv_read_reverse = GENConv(1, indim, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gin':
            self.conv_read_reverse = GINConv(MLP([1, indim], act = 'tanh', drop=opt.drop, bias= True))
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x, edge_index, edge_attr, pcd, eyes, batch=None):
                
        h, batch = self.encoder(x, edge_index, edge_attr, eyes, batch)
        h_disentangle = self.disentangle(h)
        h_disentangle = F.sigmoid(h_disentangle)

        y_pred = self.target_predict(h)
        if self.task == 'classfication':
            y_pred = F.softmax(y_pred,-1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        z1 = self.reparameterize(mu, log_var)

        z = self.z_decode(z1)
        z = z.view(z.shape[0]*z.shape[1], -1)
        x = self.conv_read_reverse(z, edge_index)

        x = self.decoder(x, edge_index, eyes, batch)
        return x, h_disentangle, y_pred, mu, log_var

class Encoder2(Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, opt, more_layer):
        super(Encoder2, self).__init__()
        self.depth = opt.depth
        if opt.act:
            self.act = F.leaky_relu
        self.down_convs = torch.nn.ModuleList()
        if opt.conv == 'gat':
            self.down_convs.append(GATConv(opt.in_channels, opt.hidden_channels, improved=True, concat=False, dropout=opt.drop, heads=4))
            if more_layer:
                self.down_convs.append(GATConv(opt.hidden_channels, opt.hidden_channels//2, improved=True, concat=False, dropout=opt.drop, heads=4))
        elif opt.conv == 'edgeconv':
            self.down_convs.append(EdgConv(opt.in_channels, opt.hidden_channels, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.down_convs.append(EdgConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
                # self.down_convs.append(EdgConv(opt.hidden_channels//2, opt.hidden_channels//4, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gen':
            self.down_convs.append(GENConv(opt.in_channels, opt.hidden_channels, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.down_convs.append(GENConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gcn':
            self.down_convs.append(GCNConv(opt.in_channels, opt.hidden_channels, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.down_convs.append(GCNConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gin':
            self.down_convs.append(GINConv(MLP([opt.in_channels, opt.hidden_channels], act = 'tanh', drop=opt.drop, bias= True)))
            if more_layer:
                self.down_convs.append(GINConv(MLP([opt.hidden_channels, opt.hidden_channels//2], act = 'tanh', drop=opt.drop, bias= True)))
        if opt.conv == 'gat':
            self.conv_read = GATConv(opt.indim, 1, improved=True, concat=False, dropout=opt.drop, heads=4)
        elif opt.conv == 'edgeconv':
            self.conv_read = EdgConv(opt.indim, 1, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gcn':
            self.conv_read = GCNConv(opt.indim, 1, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gen':
            self.conv_read = GENConv(opt.indim, 1, act = 'tanh', drop=opt.drop)
        elif opt.conv == 'gin':
            self.conv_read = GINConv(MLP([opt.indim, 1], act = 'tanh', drop=opt.drop, bias= True))
        self.knn = DilatedKnnGraph(4, 2, True, 0.2)

    def forward(self, x, edge_index, edge_attr, eyes, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        for i in range(len(self.down_convs)):
            x1 = self.down_convs[i](x, edge_index)
            e_dynamic, _ = self.knn(x1, batch)#knn动态
            x2 = self.down_convs[i](x, e_dynamic)
            x = x1 + x2
            x = self.act(x)
        
        h = self.conv_read(x, edge_index)
        # h = F.sigmoid(h)
        h = h.view(-1,eyes.shape[-1])

        # h = F.dropout(h, p=0.3, training=self.training)
        return h, batch

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

            
class Decoder2(Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, opt, hidden_channels, out_channels, more_layer):
        super(Decoder2, self).__init__()
        self.sum_res = opt.sum_res
        if opt.act:
            self.act = F.leaky_relu
        self.depth = opt.depth
        self.up_convs = torch.nn.ModuleList()
        if opt.conv == 'gat':
            self.up_convs.append(GATConv(hidden_channels, hidden_channels*2, improved=True, concat=False, dropout=opt.drop, heads=4))
            if more_layer:
                self.up_convs.append(GATConv(hidden_channels*2, out_channels, improved=True, concat=False, dropout=opt.drop, heads=4))
        elif opt.conv == 'edgeconv':
            self.up_convs.append(EdgConv(hidden_channels, hidden_channels*2, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.up_convs.append(EdgConv(hidden_channels*2, out_channels, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gcn':
            self.up_convs.append(GCNConv(hidden_channels, hidden_channels*2, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.up_convs.append(GCNConv(hidden_channels*2, out_channels, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gen':
            self.up_convs.append(GENConv(hidden_channels, hidden_channels*2, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.up_convs.append(GENConv(hidden_channels*2, out_channels, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gin':
            self.up_convs.append(GINConv(MLP([hidden_channels, hidden_channels*2], act = 'tanh', drop=opt.drop, bias= True)))
            if more_layer:
                self.up_convs.append(GINConv(MLP([hidden_channels*2, out_channels], act = 'tanh', drop=opt.drop, bias= True)))

        self.act_final = F.tanh
        self.knn = DilatedKnnGraph(4, 2, True, 0.2)

    def forward(self, x, edge_index, eyes, batch = None):
        
        for i in range(len(self.up_convs)):
            x1 = self.up_convs[i](x, edge_index)
            e_dynamic, _ = self.knn(x1, batch)#knn动态
            x2 = self.up_convs[i](x, e_dynamic)
            x = x1 + x2
        # x = self.up_convs[1](x, edge_index)
        x_ = x.view(x.shape[0]//100,100,-1)
        x_t = x_.permute(0,2,1)
        x2 = torch.bmm(x_, x_t) 
        x = x2.view(-1,100)
        diag_e = x * eyes
        x = x - diag_e

        return x
    
class ContrativeNet_infomax(torch.nn.Module):
    def __init__(self, opt):
        super(ContrativeNet_infomax, self).__init__()
     
        self.HC_net = GAE3(opt, True)
        self.disorder_net = GAE3(opt, True)


        self.disentangle_hc = nn.Linear(opt.cluster, 1)
        self.disentangle_dis = nn.Linear(opt.cluster, 1)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.bias is not None:
                module.bias.data.zero_()
   
    def forward(self, x1, edge_index1, edge_attr1, pcd1, x2, edge_index2, edge_attr2, pcd2, eyes, batch):
        '''
        

        Parameters
        ----------
        x1 : disorder
            DESCRIPTION.
        x2 : HC
            DESCRIPTION.

        Returns
        -------
        mu_hc : TYPE
            DESCRIPTION.
        log_var_hc : TYPE
            DESCRIPTION.
        z_hc : TYPE
            DESCRIPTION.
        out_hc : TYPE
            DESCRIPTION.
        mu_dis : TYPE
            DESCRIPTION.
        log_var_dis : TYPE
            DESCRIPTION.
        z_dis : TYPE
            DESCRIPTION.
        out_dis : TYPE
            DESCRIPTION.
        mu_dis_hc : TYPE
            DESCRIPTION.
        log_var_dis_hc : TYPE
            DESCRIPTION.
        z_dis_hc : TYPE
            DESCRIPTION.
        out_dis_hc : TYPE
            DESCRIPTION.

        '''

        output_dis, h_dis, y_dis = self.disorder_net(x2, edge_index2, edge_attr2, pcd2, eyes, batch)
        value_dis = torch.matmul(output_dis, h_dis).squeeze()
        logits_dis = self.disentangle_dis(value_dis)
        logits_dis = F.sigmoid(logits_dis)
        output_dis_share, h_dis_share, y_pred_dis_share = self.HC_net(x2, edge_index2,
                                                                                     edge_attr2, pcd2, eyes, batch)
        value_dis_share = torch.matmul(output_dis_share, h_dis).squeeze()
        logits_dis_share= self.disentangle_hc(value_dis_share)
        logits_dis_share = F.sigmoid(logits_dis_share)
        output_hc, h_hc, y_pred_hc = self.HC_net(x1, edge_index1, edge_attr1, pcd1, eyes, batch)
        value_hc = torch.matmul(output_hc, h_dis).squeeze()
        logits_hc = self.disentangle_hc(value_hc)
        logits_hc = F.sigmoid(logits_hc)
        
        return output_dis, h_dis, y_dis, logits_dis, output_dis_share, h_dis_share, y_pred_dis_share, logits_dis_share, \
            output_hc, h_hc, y_pred_hc, logits_hc
            
class GAE3(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, opt, more_layer):
        super().__init__()
        self.in_channels = opt.in_channels
        self.hidden_channels = opt.hidden_channels
        self.sum_res = opt.sum_res
        self.task = opt.task
        
        if more_layer:
            indim = opt.hidden_channels//2
        else:
            indim = opt.hidden_channels
        opt.indim = indim
        self.encoder = Encoder3(opt, more_layer)
        # self.bn = torch.nn.BatchNorm1d(opt.in_channels)
        if 'classification' in self.task:
            self.target_predict = nn.Linear(opt.in_channels, 2)
        elif 'regression' in self.task:
            self.target_predict = nn.Linear(opt.in_channels, 1)
        # self.target_predict2 = nn.Linear(indim, 10)
        # self.target_predict3 = nn.Linear(10, 1)
        self.z_decode = nn.Linear(indim, opt.in_channels)
        if opt.conv == 'gat':
            self.conv_read = GATConv(opt.indim, 1, improved=True, concat=False, dropout=opt.drop, heads=4)
        elif opt.conv == 'edgeconv':
            # self.conv_read = EdgConv(opt.indim, 1, act = 'tanh', drop=opt.drop)
            self.conv_read = EdgConv(opt.cluster, 1, act = opt.act, drop=opt.drop)
        elif opt.conv == 'gcn':
            self.conv_read = GCNConv(opt.indim, 1, act = 'relu', drop=opt.drop)
        elif opt.conv == 'gen':
            self.conv_read = GENConv(opt.indim, 1, act = 'relu', drop=opt.drop)
        elif opt.conv == 'gin':
            self.conv_read = GINConv(MLP([opt.indim, 1], act = 'relu', drop=opt.drop, bias= True))
        # self.fc_mu = nn.Linear(opt.in_channels, indim)
        # self.fc_var = nn.Linear(opt.in_channels, indim)
    def forward(self, x, edge_index, edge_attr, pcd, eyes, batch=None):
                
        x, batch = self.encoder(x, edge_index, edge_attr, eyes, batch)
        h = self.conv_read(x, edge_index)

        h = h.view(-1,100)

        x, _ = to_dense_batch(x, batch)
        x = torch.permute(x, (0, 2, 1))
        y_pred = self.target_predict(h)
        if self.task == 'classification':
            y_pred = F.softmax(y_pred,-1)
        h = h.unsqueeze(-1)
        return x, h, y_pred
        ############### for visualizing ROI importance in classification task
        # return y_pred

class Encoder3(Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, opt, more_layer):
        super(Encoder3, self).__init__()
        self.depth = opt.depth
        if opt.act:
            self.act = F.leaky_relu
        self.down_convs = torch.nn.ModuleList()
        if opt.conv == 'gat':
            self.down_convs.append(GATConv(opt.in_channels, opt.hidden_channels, improved=True, concat=False, dropout=opt.drop, heads=4))
            if more_layer:
                self.down_convs.append(GATConv(opt.hidden_channels, opt.hidden_channels//2, improved=True, concat=False, dropout=opt.drop, heads=4))
        elif opt.conv == 'edgeconv':
            self.down_convs.append(EdgConv(opt.in_channels, opt.hidden_channels, act = opt.act, drop=opt.drop))
            if more_layer:
                # self.down_convs.append(EdgConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
                # self.down_convs.append(EdgConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
                # self.down_convs.append(EdgConv(opt.hidden_channels//2, opt.cluster, act = 'tanh', drop=opt.drop))
                self.down_convs.append(EdgConv(opt.hidden_channels, opt.cluster, act = opt.act, drop=opt.drop))
        elif opt.conv == 'gen':
            self.down_convs.append(GENConv(opt.in_channels, opt.hidden_channels, act = 'relu', drop=opt.drop))
            if more_layer:
                self.down_convs.append(GENConv(opt.hidden_channels, opt.hidden_channels//2, act = 'relu', drop=opt.drop))
        elif opt.conv == 'gcn':
            self.down_convs.append(GCNConv(opt.in_channels, opt.hidden_channels, act = 'tanh', drop=opt.drop))
            if more_layer:
                self.down_convs.append(GCNConv(opt.hidden_channels, opt.hidden_channels//2, act = 'tanh', drop=opt.drop))
        elif opt.conv == 'gin':
            self.down_convs.append(GINConv(MLP([opt.in_channels, opt.hidden_channels], act = 'tanh', drop=opt.drop, bias= True)))
            if more_layer:
                self.down_convs.append(GINConv(MLP([opt.hidden_channels, opt.hidden_channels//2], act = 'tanh', drop=opt.drop, bias= True)))
        self.knn = DilatedKnnGraph(4, 2, True, 0.2)
        self.bn1 = torch.nn.BatchNorm1d(opt.hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(opt.cluster)

    def forward(self, x, edge_index, edge_attr, eyes, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        for i in range(len(self.down_convs)):
            x1 = self.down_convs[i](x, edge_index)
            ##################
            e_dynamic, _ = self.knn(x1, batch)#knn动态
            x2 = self.down_convs[i](x, e_dynamic)

            x = x1 + x2
            #################3# when explain the GCN, without dynamic should be used. So using this sentence 
            # x = x1 
            ##############################
            x = self.act(x)

        return x, batch

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
