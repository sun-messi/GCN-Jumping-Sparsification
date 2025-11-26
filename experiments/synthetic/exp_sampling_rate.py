"""
Experiment: Layer-wise sampling rate effects on synthetic data.

This script evaluates how different sampling rates in layer 1 (p1) and 
layer 2 (p2) affect generalization performance.

Usage:
    python exp_sampling_rate.py [--p1 0.5] [--p2 0.5] [--mode 1d|2d]
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.models import GCN
from src.utils import (
    generate_edge_index,
    generate_normalized_adj_matrix,
    generate_target_data,
    split_masks,
    sampling_topk,
)


def train(model, data, A1, A2, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, A1, A2)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, A1, A2):
    model.eval()
    out = model(data.x, A1, A2)
    
    losses = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        loss = F.mse_loss(out[mask], data.y[mask])
        losses.append(float(loss.item()))
    
    return losses


def run_single_experiment(data, A1, A2, hidden_channels, device, epochs=3001, num_runs=5):
    """Run experiment with specific adjacency matrices."""
    best_test_losses = []
    
    for run in range(num_runs):
        model = GCN(data.x.shape[1], hidden_channels, data.y.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        
        min_val_loss = float('inf')
        best_test_loss = None
        
        for epoch in range(1, epochs):
            train_loss = train(model, data, A1, A2, optimizer)
            train_loss, val_loss, test_loss = test(model, data, A1, A2)
            
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_test_loss = test_loss
        
        best_test_losses.append(best_test_loss)
    
    return sum(best_test_losses) / num_runs


def run_1d_experiment(args):
    """Run 1D experiment: vary one sampling rate while fixing the other."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Constants
    N = 2000
    d = 100
    m = 20
    k = 5
    alpha = 5
    hidden_channels = 50
    
    # Generate base graph
    edge_index = generate_edge_index(N, mean_degree=200)
    A_star = generate_normalized_adj_matrix(edge_index, N)
    
    # Generate data
    X = torch.randn(N, d)
    W = torch.randn(d, m)
    V = torch.randn(k, m)
    C = torch.randn(m, k)
    y = generate_target_data(A_star, X, W, V, C, alpha)
    
    data = Data(x=X, edge_index=edge_index, y=y)
    data.train_mask, data.val_mask, data.test_mask = split_masks(N, 0.6, 0.2, 0.2)
    data = data.to(device)
    
    sampling_rates = np.arange(0.1, 1.1, 0.1)
    
    # Experiment 1: Fix p2, vary p1
    results_vary_p1 = []
    fixed_p2 = args.p2
    print(f"\nVarying p1, fixed p2={fixed_p2}")
    
    for p1 in sampling_rates:
        print(f"  p1={p1:.1f}")
        edge_index_1 = sampling_topk(edge_index, p1)
        edge_index_2 = sampling_topk(edge_index, fixed_p2)
        A1 = generate_normalized_adj_matrix(edge_index_1, N).to(device)
        A2 = generate_normalized_adj_matrix(edge_index_2, N).to(device)
        
        avg_loss = run_single_experiment(data, A1, A2, hidden_channels, device, 
                                         epochs=args.epochs, num_runs=args.num_runs)
        results_vary_p1.append(avg_loss)
    
    # Experiment 2: Fix p1, vary p2
    results_vary_p2 = []
    fixed_p1 = args.p1
    print(f"\nVarying p2, fixed p1={fixed_p1}")
    
    for p2 in sampling_rates:
        print(f"  p2={p2:.1f}")
        edge_index_1 = sampling_topk(edge_index, fixed_p1)
        edge_index_2 = sampling_topk(edge_index, p2)
        A1 = generate_normalized_adj_matrix(edge_index_1, N).to(device)
        A2 = generate_normalized_adj_matrix(edge_index_2, N).to(device)
        
        avg_loss = run_single_experiment(data, A1, A2, hidden_channels, device,
                                         epochs=args.epochs, num_runs=args.num_runs)
        results_vary_p2.append(avg_loss)
    
    # Save and plot
    os.makedirs(args.save_dir, exist_ok=True)
    results = {
        'sampling_rates': sampling_rates.tolist(),
        'vary_p1_fixed_p2': results_vary_p1,
        'vary_p2_fixed_p1': results_vary_p2,
        'fixed_p1': fixed_p1,
        'fixed_p2': fixed_p2,
    }
    np.save(os.path.join(args.save_dir, 'sampling_rate_1d.npy'), results)
    
    plot_1d_results(results, args.save_dir)
    return results


def run_2d_experiment(args):
    """Run 2D experiment: grid search over p1 and p2."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Constants
    N = 2000
    d = 100
    m = 20
    k = 5
    alpha = 5
    hidden_channels = 50
    
    # Generate base graph
    edge_index = generate_edge_index(N, mean_degree=200)
    A_star = generate_normalized_adj_matrix(edge_index, N)
    
    # Generate data
    X = torch.randn(N, d)
    W = torch.randn(d, m)
    V = torch.randn(k, m)
    C = torch.randn(m, k)
    y = generate_target_data(A_star, X, W, V, C, alpha)
    
    data = Data(x=X, edge_index=edge_index, y=y)
    data.train_mask, data.val_mask, data.test_mask = split_masks(N, 0.6, 0.2, 0.2)
    data = data.to(device)
    
    sampling_rates = np.arange(0.1, 1.1, 0.1)
    n_rates = len(sampling_rates)
    
    results_2d = np.zeros((n_rates, n_rates))
    
    for i, p1 in enumerate(sampling_rates):
        for j, p2 in enumerate(sampling_rates):
            print(f"p1={p1:.1f}, p2={p2:.1f}")
            edge_index_1 = sampling_topk(edge_index, p1)
            edge_index_2 = sampling_topk(edge_index, p2)
            A1 = generate_normalized_adj_matrix(edge_index_1, N).to(device)
            A2 = generate_normalized_adj_matrix(edge_index_2, N).to(device)
            
            avg_loss = run_single_experiment(data, A1, A2, hidden_channels, device,
                                             epochs=args.epochs, num_runs=args.num_runs)
            results_2d[i, j] = avg_loss
    
    # Save and plot
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'sampling_rate_2d.npy'), results_2d)
    
    plot_2d_results(results_2d, sampling_rates, args.save_dir)
    return results_2d


def plot_1d_results(results, save_dir):
    """Plot 1D sampling rate results."""
    plt.figure(figsize=(10, 6), dpi=150)
    
    sampling_rates = results['sampling_rates']
    
    plt.plot(sampling_rates, np.log10(results['vary_p1_fixed_p2']), 
             'o-', linewidth=3, markersize=8, 
             label=f'Vary p₁ (shallow), p₂={results["fixed_p2"]:.1f}')
    plt.plot(sampling_rates, np.log10(results['vary_p2_fixed_p1']), 
             's--', linewidth=3, markersize=8,
             label=f'Vary p₂ (deep), p₁={results["fixed_p1"]:.1f}')
    
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Sampling Rate', fontsize=14)
    plt.ylabel('Test Error (log10)', fontsize=14)
    plt.title('Layer-wise Sampling Rate Effect (Synthetic Data)', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'sampling_rate_1d.png'), dpi=300)
    plt.close()
    print(f"Plot saved to {save_dir}/sampling_rate_1d.png")


def plot_2d_results(results_2d, sampling_rates, save_dir):
    """Plot 2D heatmap and 3D surface of sampling rate results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # 2D Heatmap
    im = axes[0].imshow(results_2d, cmap='inferno_r', 
                        extent=[0.1, 1.0, 1.0, 0.1], aspect='auto')
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Test Error', fontsize=12)
    axes[0].set_xlabel('p₂ (deep layer)', fontsize=14)
    axes[0].set_ylabel('p₁ (shallow layer)', fontsize=14)
    axes[0].set_title('2D Heatmap of Test Error', fontsize=14)
    
    # Remove 3D subplot and create new figure for 3D
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sampling_rate_2d_heatmap.png'), dpi=300)
    plt.close()
    
    # 3D Surface plot
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(sampling_rates, sampling_rates)
    surf = ax.plot_surface(X, Y, results_2d, cmap='inferno_r', linewidth=0, antialiased=True)
    
    ax.set_xlabel('p₁', fontsize=12)
    ax.set_ylabel('p₂', fontsize=12)
    ax.set_zlabel('Test Error', fontsize=12)
    ax.set_title('3D Surface of Test Error', fontsize=14)
    ax.view_init(elev=25, azim=45)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sampling_rate_2d_surface.png'), dpi=300)
    plt.close()
    
    print(f"Plots saved to {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sampling rate experiment')
    parser.add_argument('--mode', type=str, default='1d', choices=['1d', '2d'],
                        help='Experiment mode: 1d or 2d')
    parser.add_argument('--p1', type=float, default=0.3, help='Fixed p1 when varying p2')
    parser.add_argument('--p2', type=float, default=0.3, help='Fixed p2 when varying p1')
    parser.add_argument('--epochs', type=int, default=3001, help='Number of training epochs')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of experiment runs')
    parser.add_argument('--save_dir', type=str, default='results/synthetic', 
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.mode == '1d':
        run_1d_experiment(args)
    else:
        run_2d_experiment(args)
