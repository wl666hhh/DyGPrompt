import numpy as np
import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round
#名为 RandEdgeSampler，用于随机采样边（即负样本）的工具。
class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed #用指定种子创建np.random.RandomState对象，并存在self.random_state中（用于可控的随机操作）。
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)
class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = [] #邻居节点编号 (node_to_neighbors)
    self.node_to_edge_idxs = [] #对应边的索引 (node_to_edge_idxs)
    self.node_to_edge_timestamps = [] #对应边的时间戳 (node_to_edge_timestamps)

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # 我们要按照时间戳排序。
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform#一个布尔值，是否采用均匀采样（默认为False)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    从交互图中提取给定用户（源节点）在指定时间 cut_time 之前的所有交互，并按时间戳进行排序

    Returns 3 lists: neighbors, edge_idxs, timestamps 该方法返回三个列表：neighbors: 与源节点发生交互的邻居节点。edge_idxs: 与源节点的邻居之间的交互索引。timestamps: 每个交互的时间戳
    """
    #用于查找给定值 cut_time 应该插入的索引位置，保持数组的有序性。在这个情况下，i 是 cut_time 在时间戳数组中的位置。该函数会返回一个整数 i，表示所有小于 cut_time 的交互的数量
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    它根据节点 ID 列表和相应的时间戳列表，提取每个源节点的时间邻居（即与源节点在给定时间戳之前进行过交互的邻居节点）。方法支持两种采样方式：均匀采样和最近交互采样

    Params
    ------
    src_idx_l: List[int] 源节点的 ID 列表（批次大小）
    cut_time_l: List[float], 每个源节点的时间戳（交互的截止时间）
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1 #设置邻居数量。如果 n_neighbors 大于 0，则使用 n_neighbors，否则默认为 1。
    # 这段代码用于初始化三个矩阵，分别存储源节点的邻居信息、边的时间戳和交互的索引  每个矩阵的每个位置代表一个源节点与其邻居之间的交互信息。这些矩阵会在后续处理中更新，表示每个源节点与邻居之间的交互情况，且交互会根据时间戳进行排序。
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # 该矩阵的每个元素 neighbors[i, j] 表示第 i 个源节点与第 j 个邻居之间的交互。

    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # 该矩阵的每个元素 edge_times[i, j] 表示源节点 i 与其邻居 j 之间交互的时间戳。
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # 该矩阵的每个元素 edge_idxs[i, j] 表示源节点 i 与邻居 j 之间交互的索引。

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # 如果我们采用均匀抽样，则在抽样前对上面的数据进行洗牌
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # 最近交互采样
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)
          #将最近的交互（即最后 n_neighbors 次交互）填充到 neighbors、edge_times 和 edge_idxs 矩阵中的相应位置。
          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
          #i: 这是当前源节点的索引。表示要为第 i 个源节点填充邻居信息
    return neighbors, edge_idxs, edge_times
def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx#保证邻居列表可以包含所有可能的节点
  adj_list = [[] for _ in range(max_node_idx + 1)] #构造空的邻居列表（邻接表）：每个节点一个空list
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))#某个节点的每个邻居都附带交互是哪一条边、发生的时间点无向图

  return NeighborFinder(adj_list, uniform=uniform) #用前面构造的邻接表 adj_list 和 uniform 采样布尔参数实例化并返回 NeighborFinder 对象

class CustomPreLoss(torch.nn.Module):
    def __init__(self, tau=1.0):
      super(CustomPreLoss, self).__init__()
      self.tau = tau

    def forward(self,h_vt, h_at, h_bt,device='cpu'):
      # 假设 h_vt, h_at, h_bt 都是模型的输出，形状为 (batch_size, embedding_dim)
      h_vt = h_vt.to(device)
      h_at = h_at.to(device)
      h_bt = h_bt.to(device)
      # 计算相似度，使用余弦相似度
      sim_v_a = F.cosine_similarity(h_vt, h_at)
      sim_v_b = F.cosine_similarity(h_vt, h_bt)

      # 温度缩放
      scaled_sim_v_a = sim_v_a / self.tau
      scaled_sim_v_b = sim_v_b / self.tau

      # 计算损失
      numerator = torch.exp(scaled_sim_v_a)
      denominator = torch.exp(scaled_sim_v_b)

      loss = -torch.log(numerator / denominator).mean()
      return loss
if __name__ == '__main__':
    pass