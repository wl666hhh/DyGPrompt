import torch.nn as nn
import logging
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

from model.time_encoding import TimeEncoder

class DyGPrompt_pretrain(nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,n_heads=2,
              n_neighbors=None):
        super(DyGPrompt_pretrain, self).__init__()
        self.n_layers = n_layers  # 模型的层数
        self.n_heads = n_heads # 注意力机制头数
        self.neighbor_finder = neighbor_finder  # 一个用于查找图中节点邻居的函数
        self.device = device  # 运行模型的设备（CPU或GPU）
        self.logger = logging.getLogger(__name__)  # 用于调试和记录日志信息的日志对象
        #self.embeding=nn.Embedding(node_features.shape[0],node_features.shape[1])
        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(
            device)  # 将节点特征和边特征从 NumPy 数组转换为 PyTorch 张量，并将其移到指定的设备（CPU 或 GPU）
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.n_neighbors = n_neighbors  # 采样邻居个数
        self.time_encoder = TimeEncoder(time_dimension=self.n_node_features,device=device)  # 初始化时间编码器
        self.agg=nn.MultiheadAttention(embed_dim=self.node_raw_features.shape[1],kdim=self.node_raw_features.shape[1]+self.edge_raw_features.shape[1],vdim=self.node_raw_features.shape[1]+self.edge_raw_features.shape[1], num_heads=self.n_heads, dropout=0.1)

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
        n_samples=len(source_nodes)

        #维度为[n_samples,172]
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
        return source_node_embedding, destination_node_embedding, negative_node_embedding


    #用来计算每个数据集的头节点嵌入  尾结点嵌入  负样本嵌入
    def compute_temporal_embeddings(self,source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs,n_neighbors=20):
        n_samples = len(source_nodes) #源节点的数量（批次大小）
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])
        assert (self.n_layers >= 0)
        source_nodes_torch = torch.from_numpy(nodes).long().to(self.device)
        timestamps_torch = torch.from_numpy(timestamps).float().to(self.device)
        timestamps_neighbors_torch=torch.from_numpy(np.repeat(timestamps, n_neighbors)).float().to(self.device)
        #计算嵌入


        #1.初始化
        t = self.time_encoder(timestamps_torch,self.device)
        t_neighbors = self.time_encoder(timestamps_neighbors_torch,self.device)
        node_features = self.node_raw_features #有记忆的 下一个batch仍然会被看到
        h = node_features[source_nodes_torch, :]

        for i in range(self.n_layers):
            # 1.合并
            if i!=0:
                # 使用非原地操作更新 node_features，并避免原地修改 h
                index = source_nodes_torch.unsqueeze(1).expand(-1, h.size(1))  # 形状 [n_samples, feature_dim]
                # 执行 scatter
                node_features = node_features.scatter(0, index, h)
                # 创建新的 h 张量，而不是原地修改
                h = node_features[source_nodes_torch].clone()  # 关键修改：去掉 inplace 操作

            #2.Fuse
            Fuse_h_t=self.Fuse(h,t)
            #3.寻找邻居特征向量+边的特征向量
            neighbors,edge_idxs, edge_times= self.neighbor_finder.get_temporal_neighbor(
                nodes,
                timestamps,
                n_neighbors=n_neighbors)
            neighbors=neighbors.reshape(-1)
            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)  # (batch_size, n_neighbors)
            neighbors_torch=torch.from_numpy(neighbors).long().to(self.device)
            hneighbor=node_features[neighbors_torch, :]
            edge_features = self.edge_raw_features[edge_idxs, :]  # 获取边的特征

            mask=torch.from_numpy(neighbors).long().reshape(-1,n_neighbors).to(self.device)
            neighbors_padding_mask= mask == 0
            invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
            neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False


            #4.Fuse
            Fuse_hneighbor_t = self.Fuse(hneighbor, t_neighbors)
            #5.aggr
            Fuse_hneighbor_t=Fuse_hneighbor_t.view(-1, n_neighbors, node_features.shape[1])
            h=self.Aggr(Fuse_h_t,Fuse_hneighbor_t,edge_features,neighbors_padding_mask)

        #返回 [batch,n_source_nodes] [batch,n_destination_nodes] [batch,n_negative_nodes]
        return h[:n_samples],h[n_samples:2*n_samples],h[2*n_samples:]



    #融合特征函数
    def Fuse(self,h,t):
        assert h.shape == t.shape, f"Fuse函数的h和t维度不匹配 {h.shape} vs {t.shape}"
        return h+t

    #聚合函数
    # Fuse_h_t:[n_samples,172]
    # Fuse_hneighbor_t[n_samples,n_neighbors,172]
    def Aggr(self,Fuse_h_t,Fuse_hneighbor_t,edge_features,mask):

        # 相加或拼接（这里选择concat）
        combined_features_KV = torch.cat([Fuse_hneighbor_t, edge_features], dim=-1)

        Q = Fuse_h_t.unsqueeze(0)  # (1, n_samples, feature_dim) 作为查询 (Q)
        K = combined_features_KV.permute(1, 0, 2)  # (n_neighbors, n_samples, feature_dim) 作为键 (K)
        V = combined_features_KV.permute(1, 0, 2)  # (n_neighbors, n_samples, feature_dim) 作为值 (V)


        # 使用多头注意力机制+掩码
        attn_output, attn_output_weights = self.agg(Q, K, V,key_padding_mask=mask)

        # attn_output 的形状将是 [seq_len, batch_size, feature_dim]

        # 对输出结果进行调整，得到最终的输出
        attn_output = attn_output.squeeze(0)  # 去掉 seq_len 维度，得到 (n_samples, feature_dim)


        return attn_output

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder





