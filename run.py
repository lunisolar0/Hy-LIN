#!/usr/bin/env python3
"""
Hy-LIN: Hybrid Latent Interaction Network for Bug Triage.
Main entry point for training and evaluation.

Usage:
    python run.py --dataset gc         # Run on Chromium dataset
    python run.py --dataset mf         # Run on Mozilla Firefox dataset
    python run.py --dataset mc         # Run on Mozilla Core dataset
    python run.py --dataset gc --epochs 50 --K 3   # Custom settings
"""
import argparse
import os
import sys
import time
import warnings
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import prepare_dataset
from src.model import HyLIN
from src.train import bpr_loss, evaluate_model, print_metrics

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Hy-LIN Bug Triage Model')

    # Dataset
    parser.add_argument('--dataset', type=str, default='gc',
                        choices=['gc', 'mc', 'mf'],
                        help='Dataset name')

    # Model hyperparameters
    parser.add_argument('--K', type=int, default=5,
                        help='Top-K candidates per bug for augmentation')
    parser.add_argument('--L_e', type=int, default=3,
                        help='Number of explicit propagation layers')
    parser.add_argument('--L_i', type=int, default=2,
                        help='Number of implicit propagation layers')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Temperature for soft Top-K')
    parser.add_argument('--alpha_init', type=float, default=0.1,
                        help='Initial value for profile fusion alpha')
    parser.add_argument('--reg_lambda', type=float, default=1e-4,
                        help='L2 regularization coefficient')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--bpr_samples', type=int, default=16384,
                        help='Number of BPR triplets per epoch (for memory)')
    parser.add_argument('--min_bugs_per_dev', type=int, default=5,
                        help='Minimum bugs per developer to include')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Text embedding dimension (for TF-IDF+SVD)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for similarity computation')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--noise_std', type=float, default=0.10,
                        help='Std of input embedding noise for robustness '
                             '(Gaussian data augmentation)')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # ------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------
    data = prepare_dataset(
        dataset_name=args.dataset,
        min_bugs_per_dev=args.min_bugs_per_dev,
        seed=args.seed,
        emb_dim=args.emb_dim,
    )

    X_b = torch.FloatTensor(data['X_b']).to(device)
    train_adj_sp = data['train_adj']
    val_adj_sp = data['val_adj']
    test_adj_sp = data['test_adj']
    train_devs = torch.LongTensor(data['train_devs']).to(device)
    num_bugs = data['num_bugs']
    num_devs = data['num_devs']
    emb_dim = data['emb_dim']

    # Input embedding augmentation for robustness
    if args.noise_std > 0:
        noise = torch.randn_like(X_b) * args.noise_std
        X_b = X_b + noise
        print(f'Applied input augmentation (std={args.noise_std:.4f})')

    print(f'\nData stats: {num_bugs} bugs, {num_devs} developers, emb_dim={emb_dim}')

    # ------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------
    model = HyLIN(
        num_bugs=num_bugs,
        num_devs=num_devs,
        emb_dim=emb_dim,
        K=args.K,
        L_e=args.L_e,
        L_i=args.L_i,
        tau=args.tau,
        alpha_init=args.alpha_init,
        reg_lambda=args.reg_lambda,
    ).to(device)

    print(f'\nModel: {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'  Dev embeddings: {num_devs * emb_dim:,}')
    print(f'  Global proto: {emb_dim}')
    print(f'  Scalars (α, ρ, w, b): 4')

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    best_val_mrr = 0.0
    best_state = None

    print(f'\n{"="*60}')
    print(f'Training Hy-LIN on {args.dataset.upper()} ({args.epochs} epochs)')
    print(f'{"="*60}')

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # Forward + BPR loss
        model.train()
        optimizer.zero_grad()

        loss = bpr_loss(model, X_b, train_adj_sp, train_devs, device,
                        num_negatives=1, num_samples=args.bpr_samples)

        loss.backward()
        optimizer.step()

        elapsed = time.time() - t_start

        # Log
        log_str = f'[Epoch {epoch:3d}] Loss={loss.item():.4f} | Time={elapsed:.1f}s'
        if device.type == 'cuda':
            mem = torch.cuda.max_memory_allocated() / 1024**3
            log_str += f' | GPU={mem:.1f}GB'

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate_model(
                model, X_b, train_adj_sp, train_devs, val_adj_sp, test_adj_sp,
                device, ks=[1, 3, 5, 10]
            )
            log_str += '\n' + ' ' * 10
            log_str += f'val MRR={metrics["val"]["MRR"]:.4f} '
            log_str += f'HR@1={metrics["val"]["HR@1"]:.4f} '
            log_str += f'NDCG@5={metrics["val"]["NDCG@5"]:.4f}'
            log_str += ' | ' + f'test MRR={metrics["test"]["MRR"]:.4f} '
            log_str += f'HR@1={metrics["test"]["HR@1"]:.4f} '
            log_str += f'NDCG@5={metrics["test"]["NDCG@5"]:.4f}'

            # Track best
            if metrics['val']['MRR'] > best_val_mrr:
                best_val_mrr = metrics['val']['MRR']
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_metrics = metrics

        print(log_str)

    # ------------------------------------------------------------
    # Final evaluation with best model
    # ------------------------------------------------------------
    print(f'\n{"="*60}')
    print('Final Evaluation (Best Val Model)')
    print(f'{"="*60}')

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate_model(
        model, X_b, train_adj_sp, train_devs, val_adj_sp, test_adj_sp,
        device, ks=[1, 3, 5, 10]
    )
    print_metrics(final_metrics)


if __name__ == '__main__':
    main()
