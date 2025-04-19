import torch
import torch.nn as nn
import numpy as np

class DyGPrompt_downstream(nn.Module):
    def __init__(self, pretrain_model,device,bottleneck_ratio=2):
        super().__init__()
        self.pretrain_model = pretrain_model #导入预训练模型
        self.n_feat = pretrain_model.n_node_features #结点维度
        self.device=device
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

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times,edge_idxs, n_neighbors):

        #把模型中的node_raw_features变成处理好的，再提前预设好时间步

        timestamps = np.concatenate([edge_times, edge_times, edge_times])
        timestamps_torch = torch.from_numpy(timestamps).float().to(self.device)

        node_features_finetuning = self.pretrain_model.get_node_features().clone()
        time_features_finetuning = self.pretrain_model.time_encoder(timestamps_torch,self.device)

        node_features_finetuning = self.node_prompt * node_features_finetuning
        time_features_finetuning = self.time_prompt * time_features_finetuning

        conditioned_node_prompt=self.node_cond_net(node_features_finetuning)
        conditioned_time_prompt=self.time_cond_net(time_features_finetuning)

        node_features_finetuning = conditioned_time_prompt * node_features_finetuning
        time_features_finetuning = conditioned_node_prompt * time_features_finetuning

        return self.pretrain_model.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes,
                                                            edge_times, edge_idxs, n_neighbors,True,node_features_finetuning,time_features_finetuning,self.time_prompt,self.node_cond_net)




