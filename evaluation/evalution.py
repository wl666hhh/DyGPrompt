import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import torch.nn.functional as F

def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      h_vt, h_at, h_bt = model(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)
      pos_prob=torch.sigmoid(F.cosine_similarity(h_vt, h_at))
      neg_prob=torch.sigmoid(F.cosine_similarity(h_vt, h_bt))

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)

@torch.no_grad()
def eval_edge_prediction_auc(model,
                              negative_edge_sampler,
                              data,
                              n_neighbors,
                              batch_size: int = 200):
    """
    仅评估 AUC‑ROC。

    参数
    ----
    model : nn.Module
        边预测模型，前向接口为
        model(src, dst, neg_nodes, ts, eidx, n_neighbors) → (h_src, h_pos, h_neg)
    negative_edge_sampler : Sampler
        负边采样器，需先设置 seed，并实现 .reset_random_state() 与 .sample()
    data : Data
        包含 sources / destinations / timestamps / edge_idxs
    n_neighbors : int
        每个节点采样的邻居数
    batch_size : int
        推理批大小
    """
    assert negative_edge_sampler.seed is not None, "为保证可重复性，请先设置 sampler.seed"
    negative_edge_sampler.reset_random_state()

    model.eval()

    TEST_BS = batch_size
    num_edges = len(data.sources)
    num_batches = math.ceil(num_edges / TEST_BS)

    auc_scores = []

    for k in range(num_batches):
        s, e = k * TEST_BS, min(num_edges, (k + 1) * TEST_BS)

        src_batch   = data.sources[s:e]
        dst_batch   = data.destinations[s:e]
        ts_batch    = data.timestamps[s:e]
        eidx_batch  = data.edge_idxs[s:e]

        batch_size_cur = len(src_batch)
        _, neg_batch = negative_edge_sampler.sample(batch_size_cur)

        # 前向：返回 (h_src, h_pos, h_neg)
        h_src, h_pos, h_neg = model(src_batch, dst_batch,
                                    neg_batch, ts_batch,
                                    eidx_batch, n_neighbors)

        # 余弦相似度 → Sigmoid 概率
        pos_prob = torch.sigmoid(F.cosine_similarity(h_src, h_pos))
        neg_prob = torch.sigmoid(F.cosine_similarity(h_src, h_neg))

        y_score = np.concatenate([pos_prob.cpu().numpy(),
                                  neg_prob.cpu().numpy()])
        y_true  = np.concatenate([np.ones(batch_size_cur),
                                  np.zeros(batch_size_cur)])

        auc_scores.append(roc_auc_score(y_true, y_score))

    return float(np.mean(auc_scores))


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc