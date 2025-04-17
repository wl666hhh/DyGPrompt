import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

from evaluation.evalution import eval_edge_prediction
from model.dygprompt_pretrain import DyGPrompt_pretrain
from utils.utils import RandEdgeSampler, get_neighbor_finder,EarlyStopMonitor,CustomPreLoss
from data.data_processing import get_data


torch.manual_seed(0)#设置随机种子
np.random.seed(0)#设置随机种子

### Argument and global variables
parser = argparse.ArgumentParser('DyGPrompt self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='reddit') # 数据集名
parser.add_argument('--bs', type=int, default=200, help='Batch_size') # 批量大小
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints') # checkpoint前缀
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample') # 每次采样邻居数
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer') # 注意力头数
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs') # 训练轮数
parser.add_argument('--n_layer', type=int, default=5, help='Number of network layers') # 网络层数
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # 学习率
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping') # 早停轮数
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs') # 运行次数
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability') # Dropout 概率
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use') # GPU编号
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding') # 节点嵌入维度
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding') # 时间嵌入维度
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop') # 累计多少个batch再反向传播
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory') # 是否用节点记忆机制
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')  # 嵌入模块类型
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function') ## 消息函数类型
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater') ## 节点内存更新方式
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')# 消息聚合方式
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch') # 是否在batch结尾更新内存
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')# 消息维度

parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val') #验证/测试新节点是否与训练不同
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')# 时序邻居均匀采样
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')# 是否节点特征随机
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')# # 消息里是否用目标节点嵌入
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message') # 消息里是否用源节点嵌入
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model') # 是否用DyRep模型（变种）

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs # 批大小
NUM_NEIGHBORS = args.n_degree # 邻居数量
NUM_NEG = 1 # # 负采样数
NUM_EPOCH = args.n_epoch # 轮数
NUM_HEADS = args.n_head # 注意力头数
DROP_OUT = args.drop_out # dropout概率
GPU = args.gpu # 使用的GPU号
DATA = args.data # 数据集名
NUM_LAYER = args.n_layer    # 网络层数
LEARNING_RATE = args.lr # # 学习率
NODE_DIM = args.node_dim # 节点嵌入维度
TIME_DIM = args.time_dim # 时间嵌入维度
USE_MEMORY = args.use_memory # 是否使用节点记忆
MESSAGE_DIM = args.message_dim # 消息维度

Path("./saved_models/").mkdir(parents=True, exist_ok=True)# 创建保存模型的文件夹
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)# 创建临时保存模型的文件夹
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth' # 最终模型文件名
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'# checkpoint生成名字函数

### set up logger
logging.basicConfig(level=logging.INFO) # 设置日志基础级别
logger = logging.getLogger() # 获取全局Logger
logger.setLevel(logging.DEBUG) # 设置为DEBUG级别
Path("log/").mkdir(parents=True, exist_ok=True) # 创建日志文件夹
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))# 文件日志处理器，带时间戳
fh.setLevel(logging.DEBUG) # 文件日志设为debug
ch = logging.StreamHandler() # 控制台日志
ch.setLevel(logging.WARN)  # 控制台输出为warn及以上
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # 定义日志输出格式
fh.setFormatter(formatter)# 文件使用该格式
ch.setFormatter(formatter)# 控制台同上
logger.addHandler(fh)# 日志加入这些handler
logger.addHandler(ch)
logger.info(args) # 记录参数到日志

### 获得训练集验证集测试机数据
node_features, edge_features, full_data, train_data, val_data, test_data = get_data(DATA,randomize_features=args.randomize_features)

#初始化训练邻居查找器以检索时间图
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# 知道每个结点的邻居按照时间戳来进行排序 可以用来寻找邻居
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

#进行负采样
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)#为训练集初始化负采样器，不设置随机种子，默认每次采样都会变化。
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)#为验证集初始化负采样器，并设置固定种子0，确保每次运行验证集负采样结果一致。
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)#功能：为测试集初始化负采样器，设置随机种子2


#cpu或者gpu
device_string = 'cuda:0' if torch.cuda.is_available()  else "cpu"
device = torch.device(device_string)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  dygprompt = DyGPrompt_pretrain(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_neighbors=NUM_NEIGHBORS,
            )
  dygprompt.to(device)
  dygprompt.load_state_dict(torch.load('saved_models/tgn-attn-wikipedia.pth'))
  criterion = CustomPreLoss()
  optimizer = torch.optim.Adam(dygprompt.parameters(), lr=LEARNING_RATE)
  dygprompt = dygprompt.to(device)

  num_instance = len(train_data.sources)    #是训练数据的实例数
  num_batch = math.ceil(num_instance / BATCH_SIZE) #num_batch 是每个 epoch 中的批次数量，按 BATCH_SIZE 计算

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = [] #存储验证集和新节点验证集的平均精度（AP）
  val_aps = [] # 用于记录每个 epoch 和总训练时间。
  epoch_times = []
  total_epoch_times = []
  train_losses = []#train_losses 用于记录训练损失

  early_stopper = EarlyStopMonitor(max_round=args.patience) #用于提前停止训练
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()

    #设置训练图的邻居查找器
    dygprompt.set_neighbor_finder(train_ngh_finder)
    m_loss = [] #初始化存储每个批次损失的列表 m_loss

    logger.info('start {} epoch'.format(epoch)) # 训练每个批次
    for i,k in enumerate(tqdm(range(0, num_batch, args.backprop_every))):#对每个批次执行训练。每隔 args.backprop_every 个批次才执行一次反向传播。
      loss = 0
      optimizer.zero_grad()
      if i==696:
          print("hhh")
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue
        #计算开始和结束的id
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]
        #从随机采样器中获取负样本
        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)#，根据给定的采样大小从源节点和目标节点列表中随机选择节点对，并返回这些节点对。

        with torch.no_grad():#正负样本标签
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        dygprompt = dygprompt.train()
        source_node_embedding, destination_node_embedding, negative_node_embedding = dygprompt(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(source_node_embedding, destination_node_embedding, negative_node_embedding,device)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())#记录当前 loss 到 m_loss（当前 epoch 所有 mini-batch 的 loss）

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    #### Validation
    # Validation uses the full graph
    dygprompt.set_neighbor_finder(full_ngh_finder)
    val_ap, val_auc = eval_edge_prediction(model=dygprompt,
                                             negative_edge_sampler=val_rand_sampler,
                                             data=val_data,
                                             n_neighbors=NUM_NEIGHBORS)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))
      # 将当前的验证结果、训练损失和时间等信息保存到一个文件（results_path）。使用 pickle 序列化保存的数据。

    pickle.dump({
          "val_aps": val_aps,
          "new_nodes_val_aps": new_nodes_val_aps,
          "train_losses": train_losses,
          "epoch_times": epoch_times,
          "total_epoch_times": total_epoch_times
      }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))

      #如果早停检查器（early_stopper）检测到验证精度（val_ap）在最近几个周期内没有改善，就会触发早停机制。此时，模型会恢复到 最佳周期的状态（即具有最佳性能的模型），并停止训练。
    if early_stopper.early_stop_check(val_ap):
          logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
          logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
          best_model_path = get_checkpoint_path(early_stopper.best_epoch)
          dygprompt.load_state_dict(torch.load(best_model_path))
          logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
          dygprompt.eval()
          break
    else:
          torch.save(dygprompt.state_dict(), get_checkpoint_path(epoch))

      ### Test
  dygprompt.set_neighbor_finder(full_ngh_finder)
  test_ap, test_auc = eval_edge_prediction(model=dygprompt,
                                               negative_edge_sampler=test_rand_sampler,
                                               data=test_data,
                                               n_neighbors=NUM_NEIGHBORS)
  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  torch.save(dygprompt.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')


