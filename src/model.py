"""
Hy-LIN: Hybrid Latent Interaction Network for Bug Triage.
Implements learnable graph augmentation + dual-path propagation + adaptive fusion.

Reference: Section 2 (The Hy-LIN Framework) of the Hy-LIN paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseMMFunction(torch.autograd.Function):
    """Sparse-dense mm with gradient only w.r.t. the dense input.
    Avoids materializing dense gradient for the sparse adjacency."""
    @staticmethod
    def forward(ctx, sp, dn):
        ctx.save_for_backward(sp, dn)
        return torch.sparse.mm(sp, dn)

    @staticmethod
    def backward(ctx, grad_output):
        sp, dn = ctx.saved_tensors
        # Gradient w.r.t. dense input only; skip sparse matrix gradient
        grad_dn = torch.sparse.mm(sp.t(), grad_output)
        return None, grad_dn


def sparse_dense_mm(sp, dn):
    """Sparse-dense mm that doesn't compute gradient for the sparse matrix."""
    return SparseMMFunction.apply(sp, dn)


def sparse_to_torch(sparse_mat):
    """Convert scipy sparse matrix to PyTorch sparse COO tensor."""
    coo = sparse_mat.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def sparse_norm_adj(adj_sp):
    """Compute D^{-1/2} A D^{-1/2} for normalized adjacency."""
    rowsum = torch.sparse.sum(adj_sp, dim=1).to_dense()
    d_inv_sqrt = torch.pow(rowsum + 1e-12, -0.5)
    d_inv_sqrt[rowsum == 0] = 0

    N = adj_sp.shape[0]
    idx = torch.arange(N, device=adj_sp.device)
    D_inv_sqrt = torch.sparse_coo_tensor(
        torch.stack([idx, idx]), d_inv_sqrt, (N, N)
    ).coalesce()

    return torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, adj_sp), D_inv_sqrt)


class HyLIN(nn.Module):
    """
    Hy-LIN model: LightGCN backbone + semantic graph augmentation
                   + dual-path propagation + structure-aware fusion.

    Args:
        num_bugs, num_devs: graph dimensions
        emb_dim: embedding dimension (matches pre-computed text embeddings)
        K: Top-K candidates per bug for augmentation
        L_e: explicit propagation layers
        L_i: implicit propagation layers
        tau: temperature for soft Top-K relaxation
        alpha_init: initial fusion scalar α
        reg_lambda: L2 regularization coefficient
    """

    def __init__(self, num_bugs, num_devs, emb_dim, K=5,
                 L_e=3, L_i=2, tau=0.1, alpha_init=0.1, reg_lambda=1e-4):
        super().__init__()

        self.num_bugs = num_bugs
        self.num_devs = num_devs
        self.K = K
        self.L_e = L_e
        self.L_i = L_i
        self.tau = tau
        self.reg_lambda = reg_lambda

        # H_d^{(0)}: learnable developer embeddings
        self.dev_embeddings = nn.Embedding(num_devs, emb_dim)
        nn.init.xavier_uniform_(self.dev_embeddings.weight)

        # e_μ: global prototype for cold-start developers
        self.global_proto = nn.Parameter(torch.zeros(emb_dim))
        nn.init.xavier_uniform_(self.global_proto.unsqueeze(0))

        # α: semantic/structural balance (sigmoid → [0,1])
        self.alpha_raw = nn.Parameter(torch.tensor(self._inv_sigmoid(alpha_init)))

        # β = softplus(ρ): candidate edge scaling
        self.rho = nn.Parameter(torch.tensor(0.0))

        # Structure-aware gate parameters
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _inv_sigmoid(x, eps=1e-6):
        return np.log(max(min(x, 1 - eps), eps) / (1 - max(min(x, 1 - eps), eps)))

    # ================================================================
    # Section 2.2: Learnable Graph Augmentation
    # ================================================================

    def compute_target_profiles(self, X_b, train_adj_sp):
        """
        Compute normalized target-aware developer profiles (Eq. 1, 2).
        Returns profiles_norm [N_d, d] and train_adj sparse tensor.
        """
        device = X_b.device
        train_adj = sparse_to_torch(train_adj_sp).to(device)

        dev_bug_sums = sparse_dense_mm(train_adj.T, X_b)  # [N_d, d]
        dev_bug_counts = torch.sparse.sum(train_adj, dim=0).to_dense()  # [N_d]
        safe_counts = dev_bug_counts.clamp(min=1)
        static_profiles = dev_bug_sums / safe_counts.unsqueeze(1)  # [N_d, d]

        α = torch.sigmoid(self.alpha_raw)
        target_profiles = (1 - α) * static_profiles + α * self.dev_embeddings.weight

        # Cold-start: substitute global prototype for empty static profile
        cold_mask = (dev_bug_counts == 0).float().unsqueeze(1)
        target_profiles = (1 - cold_mask) * target_profiles + \
                          cold_mask * ((1 - α) * self.global_proto.unsqueeze(0) +
                                       α * self.dev_embeddings.weight)

        profiles_norm = F.normalize(target_profiles, p=2, dim=1)
        return profiles_norm, train_adj

    def sparse_topk_selection_batched(self, X_b_norm, profiles_norm,
                                       train_devs, batch_size=4096):
        """
        Batched Top-K candidate selection with STE (Eq. 5, 6, 7).

        Args:
            X_b_norm: [N_b, d] L2-normalized bug embeddings
            profiles_norm: [N_d, d] L2-normalized developer profiles
            train_devs: [N_b] LongTensor, developer assigned in training (-1 if none)

        Returns:
            M_star_indices: [2, N_b*K] top-K bug-dev pairs
            M_star_vals: [N_b*K] STE values
            S_indices: [2, N_b*K] same pairs (for S computation)
        """
        device = X_b_norm.device
        N_b, N_d = X_b_norm.shape[0], profiles_norm.shape[0]
        K = min(self.K, N_d - 1)  # at least one dev excluded

        all_topk_idx = []
        all_ste_vals = []
        row_ids = []

        for start in range(0, N_b, batch_size):
            end = min(start + batch_size, N_b)
            b_slice = slice(start, end)
            B = end - start

            X_chunk = X_b_norm[b_slice]  # [B, d]
            S_chunk = 0.5 * (1 + X_chunk @ profiles_norm.T)  # [B, N_d]

            # Mask training developers to -inf
            devs = train_devs[b_slice]  # [B]
            valid = devs >= 0
            if valid.any():
                S_chunk[valid, devs[valid]] = -float('inf')

            # Top-K
            topk_vals, topk_idx = S_chunk.topk(K, dim=1)  # [B, K]
            theta = topk_vals[:, -1:]  # [B, 1], K-th value

            # STE: forward=hard, backward=soft sigmoid gradient
            # For memory efficiency, we detach the STE values. Gradients for α
            # flow through S_vals (computed in build_augmented_adjacency).
            soft_vals = torch.sigmoid((topk_vals - theta) / self.tau)
            ste_vals = 1.0 + soft_vals - soft_vals.detach()  # [B, K]
            ste_vals = ste_vals.detach()  # no gradient through STE

            all_topk_idx.append(topk_idx)
            all_ste_vals.append(ste_vals)
            row_ids.append(torch.arange(start, end, device=device))

        # Concatenate all batches
        topk_idx = torch.cat(all_topk_idx, dim=0)      # [N_b, K]
        ste_vals = torch.cat(all_ste_vals, dim=0)      # [N_b, K]
        row_ids = torch.cat(row_ids, dim=0)             # [N_b]

        # Build sparse index format
        M_indices = torch.stack([
            row_ids.unsqueeze(1).expand(-1, K).reshape(-1),
            topk_idx.reshape(-1)
        ], dim=0)  # [2, N_b*K]

        M_vals = ste_vals.reshape(-1)  # [N_b*K]

        return M_indices, M_vals

    def build_augmented_adjacency(self, X_b_norm, profiles_norm,
                                   M_indices, M_vals, train_adj_sp):
        """
        Build augmented adjacency Â = A + β·(S ⊙ M^*)  (Eq. 8).
        Only computes S at nonzero positions of M_star.
        Returns sparse tensor [N_b, N_d].
        """
        device = X_b_norm.device
        β = F.softplus(self.rho)

        train_adj = sparse_to_torch(train_adj_sp).to(device)
        train_idx = train_adj.indices()
        train_val = train_adj.values()

        # Compute S values at M positions
        bug_idx = M_indices[0]
        dev_idx = M_indices[1]
        cos_sim = (X_b_norm[bug_idx] * profiles_norm[dev_idx]).sum(dim=1)
        S_vals = 0.5 * (1 + cos_sim)

        cand_val = β * S_vals * M_vals

        # Concatenate with training edges
        all_idx = torch.cat([train_idx, M_indices], dim=1)
        all_val = torch.cat([train_val, cand_val])
        N_b, N_d = train_adj.shape

        return torch.sparse_coo_tensor(all_idx, all_val, (N_b, N_d)).coalesce()

    # ================================================================
    # Section 2.3: Dual-Path Propagation and Fusion
    # ================================================================

    def explicit_propagation(self, A_aug, X_b, device):
        """
        Explicit bipartite propagation (LightGCN on unified graph).
        Eq. (10), (11), (12).
        """
        N_b, N_d = A_aug.shape
        N_total = N_b + N_d

        indices = A_aug.indices()
        values = A_aug.values()
        bug_idx, dev_idx = indices[0], indices[1]

        # Build symmetric unified adjacency
        upper_rows = bug_idx
        upper_cols = dev_idx + N_b
        lower_rows = dev_idx + N_b
        lower_cols = bug_idx

        all_rows = torch.cat([upper_rows, lower_rows])
        all_cols = torch.cat([upper_cols, lower_cols])
        all_vals = torch.cat([values, values])

        A_unified = torch.sparse_coo_tensor(
            torch.stack([all_rows, all_cols]), all_vals, (N_total, N_total)
        ).coalesce()

        A_norm = sparse_norm_adj(A_unified)
        # Detach: avoid materializing dense [N_total, N_total] gradients.
        # Learnable parameters (α, dev_embeddings) receive gradients through
        # the initial H and S_vals in build_augmented_adjacency.
        A_norm = A_norm.detach()

        H = torch.cat([X_b, self.dev_embeddings.weight], dim=0)
        H_list = [H]

        for _ in range(self.L_e):
            H = sparse_dense_mm(A_norm, H)
            H_list.append(H)

        return torch.stack(H_list, dim=0).mean(dim=0)  # [N_total, d]

    def developer_projected_propagation(self, A_aug, device):
        """
        Developer-projected implicit propagation (Eq. 13, 14).
        Computes R_dev = A^T D_b^{-1} A efficiently using numpy for grouping.
        """
        N_b, N_d = A_aug.shape

        # Detach: R_dev is a structural preprocessing step; gradients flow
        # through the explicit branch and dev embeddings in implicit propagation.
        D_b = torch.sparse.sum(A_aug, dim=1).to_dense().detach().cpu().numpy()
        D_b_inv = np.where(D_b > 0, 1.0 / D_b, 0.0)

        indices = A_aug.indices().detach().cpu().numpy()
        values = A_aug.values().detach().cpu().numpy()
        bug_idx = indices[0]
        dev_idx = indices[1]

        # Build COO-to-CSR-style grouping using np.bincount
        # Sort edges by bug
        sort_order = np.argsort(bug_idx)
        bug_idx_sorted = bug_idx[sort_order]
        dev_idx_sorted = dev_idx[sort_order]
        val_sorted = values[sort_order]

        # Find start positions for each bug
        bug_starts = np.searchsorted(bug_idx_sorted, np.arange(N_b))
        bug_ends = np.searchsorted(bug_idx_sorted, np.arange(N_b), side='right')

        # Accumulate R_dev
        R_dev = np.zeros((N_d, N_d), dtype=np.float32)
        for b in range(N_b):
            start, end = int(bug_starts[b]), int(bug_ends[b])
            if start == end:
                continue
            inv_D = D_b_inv[b]
            d_idx = dev_idx_sorted[start:end]
            w = val_sorted[start:end]
            # Vectorized contribution: w * w^T / D_b
            contrib = np.outer(w, w) * inv_D
            R_dev[np.ix_(d_idx, d_idx)] += contrib

        R_dev = torch.from_numpy(R_dev).to(device)
        R_dev_sp = R_dev.to_sparse().coalesce()
        R_norm = sparse_norm_adj(R_dev_sp).detach()

        H = self.dev_embeddings.weight
        H_list = [H]

        for _ in range(self.L_i):
            H = sparse_dense_mm(R_norm, H)
            H_list.append(H)

        return torch.stack(H_list, dim=0).mean(dim=0)  # [N_d, d]

    def structure_aware_fusion(self, Z_exp, Z_imp, train_adj_sp, device):
        """
        Degree-adaptive gate fusion (Eq. 16, 17).
        High-activity devs → explicit branch; low-activity → implicit branch.
        """
        N_b = self.num_bugs
        train_adj = sparse_to_torch(train_adj_sp).to(device)
        dev_degree = torch.sparse.sum(train_adj, dim=0).to_dense()  # [N_d]

        gamma = torch.sigmoid(self.w * torch.log(dev_degree + 1) + self.b)  # [N_d]

        Z_dev_exp = Z_exp[N_b:]  # [N_d, d]
        Z_dev = gamma.unsqueeze(1) * Z_dev_exp + (1 - gamma).unsqueeze(1) * Z_imp
        Z_bug = Z_exp[:N_b]

        return Z_bug, Z_dev

    # ================================================================
    # Full Forward Pass
    # ================================================================

    def forward(self, X_b, train_adj_sp, train_devs, device, batch_size=4096):
        """
        Full Hy-LIN forward pass.

        Args:
            X_b: [N_b, d] fixed bug text embeddings
            train_adj_sp: [N_b, N_d] scipy csr_matrix, training edges
            train_devs: [N_b] LongTensor, training dev per bug (-1 if none)
        Returns:
            Z_b: [N_b, d] final bug embeddings
            Z_d: [N_d, d] final developer embeddings
        """
        X_b_norm = F.normalize(X_b, p=2, dim=1)

        # Stage 1: Target-aware profiles (Section 2.2.1)
        profiles_norm, _ = self.compute_target_profiles(X_b, train_adj_sp)

        # Stage 2: Top-K candidate selection with STE (Section 2.2.3)
        M_indices, M_vals = self.sparse_topk_selection_batched(
            X_b_norm, profiles_norm, train_devs, batch_size)

        # Stage 3: Augmented adjacency (Eq. 8)
        A_aug = self.build_augmented_adjacency(
            X_b_norm, profiles_norm, M_indices, M_vals, train_adj_sp)

        # Stage 4: Explicit bipartite propagation (Section 2.3.1)
        Z_exp = self.explicit_propagation(A_aug, X_b, device)

        # Stage 5: Implicit developer-projected propagation (Section 2.3.2)
        Z_imp = self.developer_projected_propagation(A_aug, device)

        # Stage 6: Structure-aware fusion (Section 2.3.3)
        Z_b, Z_d = self.structure_aware_fusion(Z_exp, Z_imp, train_adj_sp, device)

        return Z_b, Z_d

    def get_regularization_loss(self):
        """L2 regularization on high-dimensional parameters (Eq. 19)."""
        return self.reg_lambda * (
            self.dev_embeddings.weight.pow(2).sum() +
            self.global_proto.pow(2).sum()
        )
