import torch
import torch.nn as nn

class DyGPrompt_pretrain(nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
        super(DyGPrompt_pretrain, self).__init__()

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20, device='cpu'):
        n_samples=len(source_nodes)
        #维度为[n_samples,172]
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
        return source_node_embedding, destination_node_embedding, negative_node_embedding



    #用来计算每个数据集的头节点嵌入  尾结点嵌入  负样本嵌入
    def compute_temporal_embeddings(self,source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
        pass


