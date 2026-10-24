"""可学习图增强与双路传播。

一个前向里依次做：缺陷文本投影、阀门打分、明线在整合图上传播、
暗线在开发者同质图上传播、按训练频率门控融合、内积解码。
明线和暗线的层数分开设置，两条路都不把对方的输出当输入。
暗线的起点是开发者 ID 加上该开发者自己训练缺陷的文本池化，再在同质图上向相似开发者借画像。
融合时先把两条路的分数各自标准化，再用频率门控混合，避免高频明线的余弦幅度盖住低频。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class ExplicitLayer(nn.Module):
    """一层明线传播。真实边和阀门使用不同的线性变换。"""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.w_valve = nn.Linear(hidden, hidden, bias=False)
        self.w_owner = nn.Linear(hidden, hidden, bias=False)
        self.w_bugs = nn.Linear(hidden, hidden, bias=False)
        self.w_valve_in = nn.Linear(hidden, hidden, bias=False)
        self.bug_residual = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.w_valve.weight, std=1e-3)
        nn.init.normal_(self.w_owner.weight, std=1e-3)
        nn.init.normal_(self.w_bugs.weight, std=1e-3)
        nn.init.normal_(self.w_valve_in.weight, std=1e-3)

    def update_bug(self, h: torch.Tensor, valve_msg: torch.Tensor, owner_h: torch.Tensor | None) -> torch.Tensor:
        delta = self.w_valve(valve_msg)
        if owner_h is not None:
            delta = delta + self.w_owner(owner_h)
        return _unit(h + self.drop(delta))

    def update_dev(
        self,
        h_d: torch.Tensor,
        h_b: torch.Tensor,
        owner: torch.Tensor,
        send: torch.Tensor,
        valve_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        send_f = send.float()
        msg = h_b * send_f.unsqueeze(-1)
        summed = torch.zeros_like(h_d)
        summed.index_add_(0, owner, msg)
        counts = torch.zeros(h_d.size(0), device=h_d.device, dtype=h_d.dtype)
        counts.index_add_(0, owner, send_f)
        mean = summed / counts.clamp(min=1).unsqueeze(-1)
        mean = torch.where((counts > 0).unsqueeze(-1), mean, torch.zeros_like(mean))
        delta = self.bug_residual * mean + self.drop(self.w_bugs(mean))
        if valve_in is not None:
            delta = delta + self.drop(self.w_valve_in(valve_in))
        return _unit(h_d + delta)


class ImplicitLayer(nn.Module):
    """一层暗线传播：开发者只聚合同质图上的其他开发者。"""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden, bias=False)
        self.keep = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(dropout)
        nn.init.eye_(self.lin.weight)

    def forward(self, h: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        neigh = h[knn]
        msg = (weight.unsqueeze(-1) * neigh).sum(dim=1)
        return _unit(self.keep * h + self.drop(self.lin(msg)))


class _WeightedGather(torch.autograd.Function):
    """分块聚合近邻。反向时按块重算，不把整张 [N, K, H] 留在显存里。"""

    @staticmethod
    def forward(ctx, h_src: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(h_src, weight)
        ctx.knn = knn
        out = h_src.new_empty(knn.size(0), h_src.size(1))
        step = 4096
        for start in range(0, knn.size(0), step):
            end = min(start + step, knn.size(0))
            neigh = h_src[knn[start:end]]
            out[start:end] = (weight[start:end].unsqueeze(-1) * neigh).sum(dim=1)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h_src, weight = ctx.saved_tensors
        knn = ctx.knn
        grad_h = torch.zeros_like(h_src)
        grad_w = torch.empty_like(weight)
        step = 4096
        hidden = h_src.size(1)
        for start in range(0, knn.size(0), step):
            end = min(start + step, knn.size(0))
            neigh = h_src[knn[start:end]]
            grad = grad_out[start:end]
            grad_w[start:end] = (grad.unsqueeze(1) * neigh).sum(dim=-1)
            message = grad.unsqueeze(1) * weight[start:end].unsqueeze(-1)
            grad_h.index_add_(0, knn[start:end].reshape(-1), message.reshape(-1, hidden))
        return grad_h, None, grad_w


def aggregate_valves(h_src: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _WeightedGather.apply(h_src, knn, weight)


def hetero_inflow(
    h_b: torch.Tensor,
    dev_idx: torch.Tensor,
    weight: torch.Tensor,
    send: torch.Tensor,
    n_dev: int,
) -> torch.Tensor:
    """软的缺陷→开发者边送回开发者。查询缺陷不写回，避免把测试点写进训练画像。"""
    gated = weight * send.float().unsqueeze(1)
    msg = h_b.unsqueeze(1) * gated.unsqueeze(-1)
    out = h_b.new_zeros(n_dev, h_b.size(1))
    out.index_add_(0, dev_idx.reshape(-1), msg.reshape(-1, h_b.size(1)))
    denom = h_b.new_zeros(n_dev)
    denom.index_add_(0, dev_idx.reshape(-1), gated.reshape(-1))
    return out / denom.clamp(min=1).unsqueeze(-1)


def _zscore(scores: torch.Tensor) -> torch.Tensor:
    center = scores - scores.mean(dim=-1, keepdim=True)
    return center / scores.std(dim=-1, keepdim=True).clamp(min=1e-6)


def _mixture_logits(s_exp: torch.Tensor, s_imp: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """按开发者把门控作用在两条路各自的后验上，再取对数作为排序分数。

    每条路先在开发者之间做 softmax，高频的大余弦不会直接压过低频在暗线上的相对优势。
    """
    gate = gate.clamp(1e-4, 1.0 - 1e-4)
    log_exp = torch.log_softmax(s_exp, dim=-1)
    log_imp = torch.log_softmax(s_imp, dim=-1)
    stacked = torch.stack([gate.log().unsqueeze(0) + log_exp, (1.0 - gate).log().unsqueeze(0) + log_imp], dim=0)
    return torch.logsumexp(stacked, dim=0)


class HyLIN(nn.Module):
    def __init__(
        self,
        n_dev: int,
        text_dim: int,
        hidden: int = 128,
        n_exp: int = 2,
        n_imp: int = 2,
        dropout: float = 0.2,
        fusion: str = "zscore",
    ):
        super().__init__()
        if fusion not in {"repr", "score", "zscore", "mixture", "residual"}:
            raise ValueError(f"未知融合方式 {fusion}")
        self.fusion = fusion
        self.uniform_valve = False
        self.force_gate: str | None = None
        self.gate_mode = "freq"
        self.imp_query = "prop"
        self.valve_type = "bug"
        self.in_proj = nn.Linear(text_dim, hidden, bias=False)
        if text_dim == hidden:
            nn.init.eye_(self.in_proj.weight)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
        self.bug_refine = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.bug_refine[-1].weight)
        nn.init.zeros_(self.bug_refine[-1].bias)
        self.dev_emb = nn.Embedding(n_dev, hidden)
        nn.init.normal_(self.dev_emb.weight, std=0.01)
        self.valve = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.valve[-1].weight)
        nn.init.zeros_(self.valve[-1].bias)
        self.exp_layers = nn.ModuleList(ExplicitLayer(hidden, dropout) for _ in range(n_exp))
        self.imp_layers = nn.ModuleList(ImplicitLayer(hidden, dropout) for _ in range(n_imp))
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # 频率升高时明线权重升高。初始时约 1 条偏暗线，5 条以上转明线。
        self.freq_w = nn.Parameter(torch.tensor(1.8))
        self.freq_b = nn.Parameter(torch.tensor(-2.5))
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(0.2)))))
        # residual 融合里，暗线作为明线之上的补充，初始强度约 0.3。
        self.mix_alpha = nn.Parameter(torch.tensor(-1.05))

    def project_bugs(self, text: torch.Tensor) -> torch.Tensor:
        base = self.in_proj(text)
        return _unit(base + self.bug_refine(base))

    def valve_weights(self, h_dst: torch.Tensor, h_src: torch.Tensor, knn: torch.Tensor) -> torch.Tensor:
        if self.uniform_valve:
            return h_dst.new_full((h_dst.size(0), knn.size(1)), 1.0 / knn.size(1))
        n, _k = knn.shape
        out = h_dst.new_empty(n, knn.size(1))
        # 分块并丢掉中间激活，反向时重算，避免 Core / Chromium 把近邻张量整块留在显存里。
        step = 2048

        def score_chunk(dst_h: torch.Tensor, src_h: torch.Tensor, knn_chunk: torch.Tensor) -> torch.Tensor:
            neigh = src_h[knn_chunk]
            dst_h = dst_h.unsqueeze(1).expand_as(neigh)
            raw = self.valve(torch.cat([dst_h, neigh, dst_h * neigh], dim=-1)).squeeze(-1)
            opened = torch.sigmoid(raw)
            return opened / opened.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        for start in range(0, n, step):
            end = min(start + step, n)
            out[start:end] = torch.utils.checkpoint.checkpoint(
                score_chunk, h_dst[start:end], h_src, knn[start:end], use_reentrant=False
            )
        return out

    def propagate_explicit(
        self,
        h_b: torch.Tensor,
        h_d: torch.Tensor,
        knn_b: torch.Tensor,
        w_b: torch.Tensor,
        owner: torch.Tensor,
        send: torch.Tensor,
        h_q: torch.Tensor | None = None,
        knn_q: torch.Tensor | None = None,
        w_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hetero = self.valve_type == "hetero"
        for layer in self.exp_layers:
            if self.valve_type == "none":
                valve_msg = torch.zeros_like(h_b)
            elif hetero:
                # 异质边上传的是开发者画像（ID + 其训练缺陷的文本池化），不是另一条缺陷。
                profile = h_d + self.developer_pool(h_b, owner, send)
                valve_msg = aggregate_valves(profile, knn_b, w_b)
            else:
                valve_msg = aggregate_valves(h_b, knn_b, w_b)
            owner_h = h_d[owner] * send.unsqueeze(-1)
            h_b_new = layer.update_bug(h_b, valve_msg, owner_h)
            inflow = hetero_inflow(h_b, knn_b, w_b, send, h_d.size(0)) if hetero else None
            h_d_new = layer.update_dev(h_d, h_b, owner, send, inflow)
            if h_q is not None:
                if self.valve_type == "none":
                    query_msg = torch.zeros_like(h_q)
                elif hetero:
                    query_msg = aggregate_valves(profile, knn_q, w_q)
                else:
                    query_msg = aggregate_valves(h_b, knn_q, w_q)
                h_q = layer.update_bug(h_q, query_msg, owner_h=None)
            h_b, h_d = h_b_new, h_d_new
        return h_b, h_d, h_q

    def developer_pool(self, h_b: torch.Tensor, owner: torch.Tensor, send: torch.Tensor) -> torch.Tensor:
        """每个开发者在训练缺陷上的文本均值。这是从交互里归纳出的画像，不是外部履历。"""
        send_f = send.float()
        summed = torch.zeros_like(self.dev_emb.weight)
        summed.index_add_(0, owner, h_b * send_f.unsqueeze(-1))
        counts = torch.zeros(self.dev_emb.num_embeddings, device=h_b.device, dtype=h_b.dtype)
        counts.index_add_(0, owner, send_f)
        mean = summed / counts.clamp(min=1).unsqueeze(-1)
        return torch.where((counts > 0).unsqueeze(-1), mean, torch.zeros_like(mean))

    def propagate_implicit(
        self,
        h_b: torch.Tensor,
        owner: torch.Tensor,
        send: torch.Tensor,
        dev_knn: torch.Tensor,
        dev_w: torch.Tensor,
    ) -> torch.Tensor:
        # 暗线从「可学习 ID + 自己缺陷的文本池化」出发，再在同质图上借邻居的画像。
        h = self.dev_emb.weight + self.developer_pool(h_b, owner, send)
        for layer in self.imp_layers:
            h = layer(h, dev_knn, dev_w)
        return h

    def combine(
        self,
        query: torch.Tensor,
        h_exp: torch.Tensor,
        h_imp: torch.Tensor,
        log_freq: torch.Tensor,
        query_text: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        he = _unit(h_exp)
        hi = _unit(h_imp)
        gate_in = torch.cat([he, hi], dim=-1)
        delta = self.gate(gate_in).squeeze(-1)
        if self.gate_mode == "freq":
            delta = torch.zeros_like(delta)
        elif self.gate_mode == "bounded":
            # 表示项只能把频率先验轻轻拨动，不能把低频改成明线、高频改成暗线。
            delta = 0.5 * torch.tanh(delta)
        g = torch.sigmoid(self.freq_w * log_freq + self.freq_b + delta)
        if self.force_gate == "explicit":
            g = torch.ones_like(g)
        elif self.force_gate == "implicit":
            g = torch.zeros_like(g)
        s_exp = self.score(query, he)
        # 暗线开发者画像靠近原始文本池化，查询也用投影后的原文，而不是明线传播后的表示。
        q_imp = query if self.imp_query == "prop" or query_text is None else query_text
        s_imp = self.score(q_imp, hi)
        if self.fusion == "repr":
            logits = self.score(query, g.unsqueeze(-1) * he + (1.0 - g).unsqueeze(-1) * hi)
        elif self.fusion == "score":
            logits = g * s_exp + (1.0 - g) * s_imp
        elif self.fusion == "zscore":
            logits = g * _zscore(s_exp) + (1.0 - g) * _zscore(s_imp)
        elif self.fusion == "mixture":
            logits = _mixture_logits(s_exp, s_imp, g)
        elif self.fusion == "residual":
            alpha = F.softplus(self.mix_alpha)
            mixed = _unit(he + alpha * (1.0 - g).unsqueeze(-1) * hi)
            logits = self.score(query, mixed)
        else:
            raise ValueError(self.fusion)
        return logits, g, s_exp, s_imp

    def score(self, query: torch.Tensor, dev: torch.Tensor) -> torch.Tensor:
        q = F.normalize(query, dim=-1)
        d = F.normalize(dev, dim=-1)
        temp = self.log_temp.exp().clamp(min=0.05, max=1.0)
        return q @ d.T / temp

    def encode_train_batch(
        self,
        text: torch.Tensor,
        owner: torch.Tensor,
        knn: torch.Tensor,
        batch_idx: torch.Tensor,
        dev_knn: torch.Tensor,
        dev_w: torch.Tensor,
        log_freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练前向。当前 batch 的真实边从整合图里拿掉，避免缺陷表示直接看见自己的负责人。"""
        h0 = self.project_bugs(text)
        h_d0 = self.dev_emb.weight
        send = torch.ones(text.size(0), dtype=torch.bool, device=text.device)
        send[batch_idx] = False
        if self.valve_type == "hetero":
            src = h_d0 + self.developer_pool(h0, owner, send)
        else:
            src = h0
        w = self.valve_weights(h0, src, knn)
        h_b, h_exp, _ = self.propagate_explicit(h0, h_d0, knn, w, owner, send)
        h_imp = self.propagate_implicit(h0, owner, send, dev_knn, dev_w)
        logits, g, s_exp, s_imp = self.combine(h_b[batch_idx], h_exp, h_imp, log_freq, query_text=h0[batch_idx])
        return logits, g, w, s_exp, s_imp

    def encode_queries(
        self,
        train_text: torch.Tensor,
        train_owner: torch.Tensor,
        train_knn: torch.Tensor,
        query_text: torch.Tensor,
        query_knn: torch.Tensor,
        dev_knn: torch.Tensor,
        dev_w: torch.Tensor,
        log_freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """验证 / 测试前向。查询缺陷没有真实边，开发者表示只由训练图决定。"""
        h0 = self.project_bugs(train_text)
        q0 = self.project_bugs(query_text)
        h_d0 = self.dev_emb.weight
        send = torch.ones(train_text.size(0), dtype=torch.bool, device=train_text.device)
        if self.valve_type == "hetero":
            src = h_d0 + self.developer_pool(h0, train_owner, send)
        else:
            src = h0
        w_b = self.valve_weights(h0, src, train_knn)
        w_q = self.valve_weights(q0, src, query_knn)
        _, h_exp, h_q = self.propagate_explicit(
            h0,
            self.dev_emb.weight,
            train_knn,
            w_b,
            train_owner,
            send,
            h_q=q0,
            knn_q=query_knn,
            w_q=w_q,
        )
        h_imp = self.propagate_implicit(h0, train_owner, send, dev_knn, dev_w)
        logits, g, _s_exp, _s_imp = self.combine(h_q, h_exp, h_imp, log_freq, query_text=q0)
        return logits, g
