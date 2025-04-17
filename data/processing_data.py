import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f) #跳过表头
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0]) #入边
      i = int(e[1]) #出边

      ts = float(e[2]) #时间戳
      label = float(e[3])  # 边的标签

      feat = np.array([float(x) for x in e[4:]]) #边的特征

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx) #id

      feat_l.append(feat) #：返回Pandas DataFrame (节点ID、时间戳、标签、行号) 和特征矩阵（NumPy数组）。
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df):
  new_df = df.copy()
  assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
  assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
  #为物品编号“腾出”一块区间。假如用户共有0～999号，则upper_u为1000
  upper_u = df.u.max() + 1
  new_i = df.i + upper_u #把所有物品的编号整体平移到用户编号之后的区间

  new_df.i = new_i
  new_df.u += 1 #用户编号整体加1。 变为1～upper_u
  new_df.i += 1 #物品编号也整体加1
  new_df.idx += 1 #行号（索引）加1

  return new_df

def run(data_name):
  #创建输入输出路径
  Path("/").mkdir(parents=True, exist_ok=True)
  PATH = './{}.csv'.format(data_name)
  OUT_DF = './ml_{}.csv'.format(data_name)
  OUT_FEAT = './ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './ml_{}_node.npy'.format(data_name)
  #
  df, feat = preprocess(PATH)
  new_df = reindex(df)  #这段代码将两个集合（用户和物品）的编号“错开分区”，再统一右移一位（都加1），保证节点编号唯一、正整数、并为特殊用途预留0号

  empty = np.zeros(feat.shape[1])[np.newaxis, :]#创建一行全为0的特征，长度等于每条边的特征维数
  feat = np.vstack([empty, feat])#最终特征索引0对应“特殊”边（比如padding等），以便在模型中用索引0时不会报错 同时保证数据对齐

  max_idx = max(new_df.u.max(), new_df.i.max())#找出编号过后的节点ID的最大值
  rand_feat = np.zeros((max_idx + 1,172)) #每一行代表一个节点的特征向量

  new_df.to_csv(OUT_DF) #保存边列表（含新编号等信息）为csv文件
  np.save(OUT_FEAT, feat) #保存边特征矩阵为numpy文件
  np.save(OUT_NODE_FEAT, rand_feat) #保存初始化的节点特征矩阵

parser = argparse.ArgumentParser('Interface for DyGPrompt data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')

args = parser.parse_args()

run(args.data)

