import torch
import torch.nn as nn


class DyGPrompt_downstream(nn.Module):
    def __init__(self, pretrain_model, prompt_dim=64, bottleneck_ratio=2):
        super().__init__()
        self.pretrain_model = pretrain_model #导入预训练模型
        self.n_feat = pretrain_model.n_node_features #结点维度

        # 冻结预训练参数
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

        # 双提示向量
        self.node_prompt = nn.Parameter(torch.randn(self.n_feat))  # 节点提示
        self.time_prompt = nn.Parameter(torch.randn(self.n_feat))  # 时间提示

        # 双条件网络
        hidden_dim = self.n_feat // bottleneck_ratio
        self.time_cond_net = nn.Sequential(  # 时间条件网络生成节点提示
            nn.Linear(self.n_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_feat)
        )
        self.node_cond_net = nn.Sequential(  # 节点条件网络生成时间提示
            nn.Linear(self.n_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_feat)
        )

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times,edge_idxs, n_neighbors=20):

        #维度为[n_samples,172]
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_embeddings_with_feat(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
        return source_node_embedding, destination_node_embedding, negative_node_embedding


    def compute_embeddings_with_feat(self,source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
        """带提示机制的前向传播"""
        # 基础特征获取
        base_node_feat = self.pretrain_model.node_raw_features[source_nodes]
        time_feat = self.pretrain_model.time_encoder(timestamps, self.pretrain_model.device)

        # 基础提示调整
        node_feat = base_node_feat * self.node_prompt  # 式(4)
        adjusted_time = time_feat * self.time_prompt  # 式(5)

        # 动态条件提示
        dynamic_node_prompt = self.time_cond_net(adjusted_time)  # 式(6)
        dynamic_time_prompt = self.node_cond_net(node_feat)  # 式(8)

        # 最终特征调整
        final_node_feat = node_feat * dynamic_node_prompt
        final_time_feat = adjusted_time * dynamic_time_prompt

        # 特征融合
        fused_feat = final_node_feat + final_time_feat

        # 获取邻居信息（复用预训练模型的逻辑）
        neighbors, _, edge_times = self.pretrain_model.neighbor_finder.get_temporal_neighbor(
            source_nodes.cpu().numpy(),
            timestamps.cpu().numpy(),
            n_neighbors=n_neighbors)

        # 通过预训练编码器
        h = self.pretrain_model.compute_embeddings_with_feat(
            fused_feat,
            neighbors,
            edge_times)

        return h