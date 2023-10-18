import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json

from TemporalAttention import TemporalAttention

import ipdb

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class GCN_Layer(nn.Module):
    def __init__(self,num_of_features,num_of_filter):
        """One layer of GCN
        
        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer,self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features = num_of_features,
                    out_features = num_of_filter),
            nn.ReLU()
        )
    def forward(self,input,adj):
        """计算一层GCN
        
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)
        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size,_,_ = input.shape
        adj = torch.from_numpy(adj).to(input.device)
        adj = adj.repeat(batch_size,1,1)
        input = torch.bmm(adj, input)
        output = self.gcn_layer(input)
        return output

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__(
        self,
        in_channels,
        ch1x1,
        ch3x3red,
        ch3x3,
        ch5x5red,
        ch5x5,
        pool_proj,
        conv_block=None
    ):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), # MLP for pre-processing
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1, dilation=1)
        )
        # self.branch2 = conv_block(in_channels, ch3x3, kernel_size=3, padding=1)
        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1), #MLP
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2, dilation=1),
        )
        # self.branch3 = conv_block(in_channels, ch5x5, kernel_size=5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x):
        # ipdb.set_trace()
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        # outputs = [branch2, branch3]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class STGeoModule(nn.Module):
    def __init__(self,grid_in_channel,num_of_target_time_feature,K):
        """[summary]
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_target_time_feature {int} -- the number of target time feature: 24(hour)+7(week)+1(holiday)=32
            K {int} -- The number of the Head in Attention
        """
        super(STGeoModule,self).__init__()
        ch1x1 = 64
        ch3x3red = 64
        ch3x3 = 32
        ch5x5red= 64
        ch5x5 = 32
        pool_proj = 64
        D_cnn = ch1x1+ch3x3+ch5x5+pool_proj
        # D_cnn = ch3x3+ch5x5+ch1x1
        # self.grid_conv1 = Inception(grid_in_channel,64,96,128,16,32,32)#192, 64, 96, 128, 16, 32, 32
        # self.grid_conv2 = Inception(D_cnn,64,96,128,16,32,32)
        self.grid_conv1 = Inception(grid_in_channel,ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5,pool_proj)
        self.grid_conv2 = Inception(D_cnn,ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5,pool_proj)
        self.d_model = 41
        L = 2
        K = K
        d = 8
        self.D = K*d
        self.grid_att_fc1 = nn.Linear(in_features=self.d_model,out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)

        self.fc_start = nn.Linear(D_cnn,self.D)
        self.fc_end = nn.Linear(self.D,self.d_model)

        self.enc_geo_T = torch.nn.ModuleList([TemporalAttention(self.d_model,K,d) for _ in range(L)])
        


    def forward(self,grid_input,target_time_feature,grid_node_map):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,seq_len,D,W,H)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
        Returns:
            {Tensor} -- shape：(batch_size,hidden_size,W,H)
        """
        B,T,D,W,H = grid_input.shape
        # grid_input_new = grid_input.view(B*T,D,W*H) 
        
        # grid_node_map_tmp = torch.from_numpy(grid_node_map).to(target_time_feature.device.type).repeat(B*T,1,1)
        # grid_input_new_S = torch.bmm(grid_input_new,grid_node_map_tmp).permute(0,2,1)
        
        grid_input = grid_input.view(-1,D,W,H)
        conv_output = self.grid_conv2(self.grid_conv1(grid_input))
        D_cnn = conv_output.shape[1]
        conv_output = conv_output.view(B,-1,D_cnn,W,H)\
                        .permute(0,3,4,1,2)\
                        .contiguous()\
                        .view(-1,T,D_cnn)
        # gru_output,_ = self.grid_gru(conv_output)
        # X = conv_output
        # spataial transformer
        # reshape the BTDWH to B*T,W*H,D
        # grid_input_new = grid_input.permute(0,1,3,4,2).contiguous().view(batch_size*T,W*H,D)
        # 
        X = conv_output #D=48
        # for net in self.enc_geo_S:
        #     X = net(X) #D=48
        # X = X.view(B,T,-1,D).permute(0,2,1,3).contiguous().view(-1,T,D)
        X = self.fc_start(X)
        for net in self.enc_geo_T: 
            X = net(X) #D=48
        X = self.fc_end(X)
        # ipdb.set_trace()
        # X = X.contiguous().view(B,-1,T,D).permute(0,2,3,1).contiguous().view(B*T,D,-1) #B*T D N
        # grid_node_map_tmp B*T,W*H,N
        # grid_node_map_tmp_new = grid_node_map_tmp.permute(0,2,1) #B*T,N,W*H
        # X = torch.bmm(X,grid_node_map_tmp_new).view(B,T,D,W,H).permute(0,3,4,1,2).contiguous().view(B*W*H,T,D) # B*T,D,W*H
        grid_target_time = torch.unsqueeze(target_time_feature,1).repeat(1,W*H,1).view(B*W*H,-1)
        grid_att_fc1_output = torch.squeeze(self.grid_att_fc1(X))
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output+grid_att_fc2_output+self.grid_att_bias))
        grid_att_score = grid_att_score.view(B*W*H,-1,1)
        grid_output = torch.sum(X * grid_att_score,dim=1)
        grid_output = grid_output.view(B,W,H,-1).permute(0,3,1,2).contiguous()
    
        return grid_output


class RiskContra(nn.Module):
    def __init__(self,grid_in_channel,pre_len,
                gru_hidden_size,num_of_target_time_feature,
                num_of_graph_feature,nums_of_graph_filters,
                north_south_map,west_east_map,fusion,K):
        """[summary]
        
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            pre_len {int} -- the time length of prediction
            num_of_target_time_feature {int} -- the number of target time feature: 24(hour)+7(week)+1(holiday)=32
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data
        """
        super(RiskContra,self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map

        self.st_geo_module = STGeoModule(grid_in_channel,num_of_target_time_feature,K)

        fusion_channel = fusion
        self.grid_weigth = nn.Conv2d(in_channels=41,out_channels=fusion_channel,kernel_size=1)
        # self.graph_weigth = nn.Conv2d(in_channels=64,out_channels=fusion_channel,kernel_size=1)
        self.output_layer = nn.Linear(fusion_channel*north_south_map*west_east_map,pre_len*north_south_map*west_east_map)

        self.g1 = nn.Linear(in_features=41,out_features=41)
        # self.g2 = nn.Linear(in_features=41,out_features=41)


    def forward(self,grid_input,target_time_feature,
                road_adj,risk_adj,poi_adj,grid_node_map):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,T,D,W,H)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N,N)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)

        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size,_,_,W,H =grid_input.shape
        # ipdb.set_trace()
        grid_output = self.st_geo_module(grid_input,target_time_feature,grid_node_map)
        # graph_output = self.st_sem_module(graph_feature,road_adj,risk_adj,poi_adj,
        #                                 target_time_feature,grid_node_map)
        # add the MLP layer to get Z for the contra_loss, MLP(Relu(MLP(X)))
        grid_output_ = grid_output.permute(0,2,3,1).view(batch_size,W,H,-1)
        # grid_mlp = self.g2(F.relu(self.g1(grid_output_)))
        grid_mlp = F.relu(self.g1(grid_output_))
        grid_mlp = grid_mlp.permute(0,3,1,2).contiguous()
        grid_output = self.grid_weigth(grid_output)
        # graph_output = self.graph_weigth(graph_output)
        fusion_output = (grid_output).view(batch_size,-1)#+ graph_output
        final_output = self.output_layer(fusion_output)\
                            .view(batch_size,-1,self.north_south_map,self.west_east_map)

        return final_output,grid_mlp
