import numpy as np
import random
import pandas as pd



class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources) #本组数据有多少“边”
    self.unique_nodes = set(sources) | set(destinations) #唯一节点集合和数目
    self.n_unique_nodes = len(self.unique_nodes) #上面集合的节点数

#预训练数据集函数
def get_data(dataset_name,randomize_features=True,mode="pretrain"):
  ##
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  #预训练数据集还是微调数据集
  if mode=="pretrain":
    # 计算前 80% 数据的切分点
    split_index = int(0.80 * len(graph_df))  # 根据数据长度计算80%的切分点
    graph_df = graph_df[:split_index]  # 取前 80% 的行
    edge_features = edge_features[:split_index]  # 假设边特征的数量与数据行数相同

  if mode=="finetuning":
    # 计算后 20% 数据的切分点
    split_index = int(0.20 * len(graph_df))  # 根据数据长度计算80%的切分点
    graph_df = graph_df[split_index:]  # 取前 80% 的行
    edge_features = edge_features[split_index:]  # 假设边特征的数量与数据行数相同

  #如果要“打乱特征”，则用随机值替换节点特征
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])
  #取“时间戳”列的80%和90%分位点，分别作为训练/验证、验证/测试集的时间切分点。
  val_time, test_time = list(np.quantile(graph_df.ts, [0.80, 0.90]))
  #取每一列数据，变成numpy数组，分别表示边的起点、终点、下标、标签、时间戳。
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  #封装成自定义数据类
  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020) #固定随机种子，为可复现实验

  node_set = set(sources) | set(destinations)

  train_mask = timestamps <= val_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # 这两行用于生成标准验证集和测试集（所有的验证/测试边，不关心是不是“新节点”）。
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))


  return node_features, edge_features, full_data, train_data, val_data, test_data