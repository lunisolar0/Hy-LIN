"""图增强的“引导来源”消融（回应 C5 自证偏差）。

锁定方法用文本近邻（MiniLM top-k）作为增强候选图。这里给出两个非文本对照：

- random：每条缺陷随机连到训练缺陷。只为证明“多接一批边”本身没有用，
  真正的增益来自“按文本相似挑邻”。
- node2vec：把训练期的“缺陷–负责人”二部图做 node2vec，得到缺陷的结构向量；
  增强候选图按结构向量取近邻。查询缺陷没有边，用它的文本近邻在结构空间的均值近似。

两者都只替换增强候选图，其余保持锁定配置。
"""
from __future__ import annotations

import numpy as np
import torch


def random_knn(n_src: int, n_train: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, n_train, (n_src, k), generator=gen)


def _node2vec_vectors(edges_src: np.ndarray, edges_dst: np.ndarray, n_bug: int, dim: int = 64,
                      walk_length: int = 20, walks_per_node: int = 5, p: float = 1.0, q: float = 1.0,
                      seed: int = 42) -> np.ndarray:
    import networkx as nx
    from gensim.models import Word2Vec

    g = nx.Graph()
    n_owner = int(edges_dst.max()) + 1
    for s, d in zip(edges_src, edges_dst):
        g.add_edge(int(s), int(n_bug + d))
    nodes = list(g.nodes())
    rng = np.random.default_rng(seed)
    adj = {n: list(g.neighbors(n)) for n in nodes}

    def walk(start: int) -> list[int]:
        path = [start]
        while len(path) < walk_length:
            cur = path[-1]
            nbrs = adj[cur]
            if not nbrs:
                break
            if len(path) == 1:
                nxt = nbrs[rng.integers(len(nbrs))]
            else:
                prev = path[-2]
                weights = []
                for nb in nbrs:
                    if nb == prev:
                        weights.append(1.0 / p)
                    elif g.has_edge(nb, prev):
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / q)
                w = np.asarray(weights, dtype=np.float64)
                w = w / w.sum()
                nxt = nbrs[rng.choice(len(nbrs), p=w)]
            path.append(nxt)
        return [str(x) for x in path]

    corpus = []
    for n in nodes:
        for _ in range(walks_per_node):
            corpus.append(walk(n))
    model = Word2Vec(corpus, vector_size=dim, window=5, min_count=0, sg=1, workers=8, epochs=3, seed=seed)
    vec = np.zeros((n_bug, dim), dtype=np.float32)
    for b in range(n_bug):
        key = str(b)
        if key in model.wv:
            vec[b] = model.wv[key]
    return vec


def node2vec_knn(bundle: dict, k: int, seed: int, dim: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_dev = int(bundle["n_dev"])
    train_idx = bundle["train_idx"].numpy()
    owner = bundle["owners"].numpy()[train_idx]
    n_train = int(train_idx.size)
    # 训练缺陷局部编号 0..n_train-1
    vec = _node2vec_vectors(np.arange(n_train), owner, n_train, dim=dim, seed=seed)
    norm = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-8)
    vn = vec / norm

    def knn_self(src: np.ndarray, exclude_self: bool) -> torch.Tensor:
        sim = torch.from_numpy(src @ vn.T)
        kk = k + 1 if exclude_self else k
        if exclude_self:
            rows = torch.arange(sim.size(0))
            sim[rows, rows] = -1e9
        return torch.topk(sim, kk, dim=-1).indices[:, :k]

    train_knn = knn_self(vn, True)
    # 查询缺陷：用其文本近邻在结构空间的均值近似结构向量
    text_knn = bundle["train_knn"].numpy()
    for split in ("val_knn", "test_knn"):
        pass
    def query_vec(text_knn_rows: torch.Tensor) -> np.ndarray:
        idx = text_knn_rows.numpy()
        return vn[idx].mean(axis=1)

    val_knn = knn_self(query_vec(bundle["val_knn"]), False)
    test_knn = knn_self(query_vec(bundle["test_knn"]), False)
    return train_knn, val_knn, test_knn
