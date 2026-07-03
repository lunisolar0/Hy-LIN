"""
Training utilities: BPR loss, negative sampling, evaluation metrics.
"""
import torch
import torch.nn.functional as F
import numpy as np


def sample_bpr_triplets(train_adj_sp, num_negatives=1):
    """
    Sample BPR training triplets (b_i, d_j+, d_j-) from training edges.
    """
    train_coo = train_adj_sp.tocoo()
    num_pos = train_coo.nnz
    N_d = train_adj_sp.shape[1]

    bug_ids = np.repeat(train_coo.row, num_negatives)
    pos_ids = np.repeat(train_coo.col, num_negatives)
    neg_ids = np.random.randint(0, N_d, size=len(bug_ids))

    train_csr = train_adj_sp.tocsr()
    for i in range(len(bug_ids)):
        b = bug_ids[i]
        row_start = train_csr.indptr[b]
        row_end = train_csr.indptr[b + 1]
        pos_set = set(train_csr.indices[row_start:row_end])
        while neg_ids[i] in pos_set or neg_ids[i] == pos_ids[i]:
            neg_ids[i] = np.random.randint(0, N_d)

    return bug_ids, pos_ids, neg_ids


def bpr_loss(model, X_b, train_adj_sp, train_devs, device,
             num_negatives=1, num_samples=None):
    """
    Compute BPR loss (Eq. 18) on sampled triplets.

    Args:
        num_samples: if set, randomly sample this many training edges for BPR.
                     If None, uses all training edges.
    """
    Z_b, Z_d = model(X_b, train_adj_sp, train_devs, device)

    bug_ids, pos_ids, neg_ids = sample_bpr_triplets(train_adj_sp, num_negatives)

    # Sub-sample if requested
    if num_samples is not None and num_samples < len(bug_ids):
        idx = np.random.choice(len(bug_ids), num_samples, replace=False)
        bug_ids = bug_ids[idx]
        pos_ids = pos_ids[idx]
        neg_ids = neg_ids[idx]

    bug_ids = torch.LongTensor(bug_ids).to(device)
    pos_ids = torch.LongTensor(pos_ids).to(device)
    neg_ids = torch.LongTensor(neg_ids).to(device)

    pos_scores = (Z_b[bug_ids] * Z_d[pos_ids]).sum(dim=1)
    neg_scores = (Z_b[bug_ids] * Z_d[neg_ids]).sum(dim=1)

    loss_bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
    loss_reg = model.get_regularization_loss()

    return loss_bpr + loss_reg


def evaluate_model(model, X_b, train_adj_sp, train_devs, val_adj_sp,
                   test_adj_sp, device, ks=(1, 3, 5, 10)):
    """
    Evaluate on validation and test sets.
    Computes NDCG@K, HR@K, and MRR.
    """
    model.eval()
    with torch.no_grad():
        Z_b, Z_d = model(X_b, train_adj_sp, train_devs, device)

    scores = Z_b @ Z_d.T  # [N_b, N_d]

    metrics = {}
    for name, adj_sp in [('val', val_adj_sp), ('test', test_adj_sp)]:
        adj_coo = adj_sp.tocoo()
        eval_bugs = np.unique(adj_coo.row)

        hr = {k: [] for k in ks}
        ndcg = {k: [] for k in ks}
        mrr_list = []

        for b in eval_bugs:
            gt_devs = set(adj_coo.col[adj_coo.row == b])
            if len(gt_devs) == 0:
                continue

            bug_scores = scores[b].cpu().numpy()
            ranking = np.argsort(-bug_scores)

            for k in ks:
                top_k = set(ranking[:k])
                hit = 1 if len(top_k & gt_devs) > 0 else 0
                hr[k].append(hit)

            for k in ks:
                dcg = 0.0
                for rank, dev in enumerate(ranking[:k]):
                    if dev in gt_devs:
                        dcg += 1.0 / np.log2(rank + 2)
                idcg = 1.0 / np.log2(2)
                for r in range(1, min(k, len(gt_devs))):
                    idcg += 1.0 / np.log2(r + 2)
                ndcg_val = dcg / idcg if idcg > 0 else 0.0
                ndcg[k].append(ndcg_val)

            for rank, dev in enumerate(ranking):
                if dev in gt_devs:
                    mrr_list.append(1.0 / (rank + 1))
                    break
            else:
                mrr_list.append(0.0)

        name_metrics = {}
        for k in ks:
            name_metrics[f'HR@{k}'] = np.mean(hr[k])
            name_metrics[f'NDCG@{k}'] = np.mean(ndcg[k])
        name_metrics['MRR'] = np.mean(mrr_list)
        metrics[name] = name_metrics

    model.train()
    return metrics


def print_metrics(metrics, epoch=None):
    """Pretty-print evaluation metrics."""
    prefix = f'[Epoch {epoch:3d}]' if epoch is not None else '[Final]'
    for split in ['val', 'test']:
        if split not in metrics:
            continue
        m = metrics[split]
        hr_str = ' '.join([f'HR@{k}={m[f"HR@{k}"]:.4f}' for k in [1, 3, 5, 10]])
        ndcg_str = ' '.join([f'NDCG@{k}={m[f"NDCG@{k}"]:.4f}' for k in [1, 3, 5, 10]])
        mrr_str = f'MRR={m["MRR"]:.4f}'
        print(f'{prefix} {split:>4s} | {hr_str} | {ndcg_str} | {mrr_str}')
