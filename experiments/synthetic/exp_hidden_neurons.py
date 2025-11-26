"""
Experiment: Test error vs. hidden neurons under different adjacency matrix norms.

This script reproduces results from Section 4.1, analyzing how the generalization
error depends on the number of hidden neurons and the 1-norm of A*.

Usage:
    python exp_hidden_neurons.py [--save_dir SAVE_DIR]
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

from src.models import GCN_PyG
from src.utils import (
    generate_sbm_edge_index,
    generate_normalized_adj_matrix,
    generate_target_data,
    split_masks,
)


def train(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, device):
    model.eval()
    out = model(data.x, data.edge_index)
    
    losses = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        loss = F.mse_loss(out[mask], data.y[mask])
        losses.append(float(loss.item()))
    
    return losses


def run_experiment(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Constants
    N = 2000
    N1 = 300  # Number of nodes in group 1
    d = 100   # Feature dimension
    m = 20    # Intermediate dimension
    k = 5     # Output dimensions
    alpha = 5
    
    hidden_channel_options = [5, 10, 20, 50, 100]
    configs = [(200, 200), (200, 150), (200, 120)]  # (d1, d2) configurations
    
    num_runs = args.num_runs
    final_test_losses_by_config = {}
    
    for d1, d2 in configs:
        print(f"\nRunning experiment with d1 = {d1}, d2 = {d2}")
        
        avg_final_test_losses = np.zeros(len(hidden_channel_options))
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            # Generate graph
            p1 = d1 / N
            p2 = d2 / N
            edge_index = generate_sbm_edge_index(N, p1, p2, N1)
            
            A_star = generate_normalized_adj_matrix(edge_index, N)
            one_norm = torch.norm(A_star, p=1, dim=0).max()
            
            # Generate features and labels
            X = torch.randn(N, d)
            W = torch.randn(d, m)
            V = torch.randn(k, m)
            C = torch.randn(m, k)
            y = generate_target_data(A_star, X, W, V, C, alpha)
            
            # Create data object
            data = Data(x=X, edge_index=edge_index, y=y)
            data.train_mask, data.val_mask, data.test_mask = split_masks(N, 0.6, 0.2, 0.2)
            data = data.to(device)
            
            # Train and evaluate for each hidden channel size
            final_test_losses = []
            for hidden_channels in hidden_channel_options:
                model = GCN_PyG(X.shape[1], hidden_channels, y.shape[1]).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
                
                min_val_loss = float('inf')
                best_test_loss = None
                
                for epoch in range(1, args.epochs + 1):
                    train_loss = train(model, data, optimizer, device)
                    train_loss, val_loss, test_loss = test(model, data, device)
                    
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_test_loss = test_loss
                
                final_test_losses.append(best_test_loss)
            
            avg_final_test_losses += np.array(final_test_losses)
        
        avg_final_test_losses /= num_runs
        final_test_losses_by_config[float(one_norm)] = avg_final_test_losses.tolist()
        print(f"  ||A*||_1 = {one_norm:.2f}, Test losses: {avg_final_test_losses}")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'final_test_losses_by_config.npy'), final_test_losses_by_config)
    
    # Plot results
    plot_results(final_test_losses_by_config, hidden_channel_options, args.save_dir)
    
    return final_test_losses_by_config


def plot_results(results, hidden_channel_options, save_dir):
    """Plot test error vs hidden neurons for different A* norms."""
    markers = ['o', 's', 'D']
    colors = ['b', 'g', 'r']
    linewidths = [3, 3, 3]
    
    plt.figure(figsize=(8, 6), dpi=150)
    
    for index, (one_norm, losses) in enumerate(results.items()):
        plt.plot(
            hidden_channel_options, 
            np.log10(losses), 
            marker=markers[index % len(markers)],
            markersize=10,
            color=colors[index % len(colors)],
            linewidth=linewidths[index % len(linewidths)],
            label=r'$\|A^*\|_1=$' + f'{one_norm:.2f}'
        )
    
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.xscale('log')
    plt.xlabel('Total number of hidden neurons', fontsize=14)
    plt.ylabel('Test error (log10)', fontsize=14)
    plt.title('Test Error vs. Hidden Neurons', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'hidden_neurons_vs_error.png'), dpi=300)
    plt.close()
    print(f"Plot saved to {save_dir}/hidden_neurons_vs_error.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hidden neurons experiment')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of experiment runs')
    parser.add_argument('--save_dir', type=str, default='results/synthetic', help='Directory to save results')
    
    args = parser.parse_args()
    run_experiment(args)
