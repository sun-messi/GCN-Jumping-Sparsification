"""
Visualization scripts for plotting experiment results.

This module provides functions to:
- Load saved experiment results
- Generate publication-quality figures
- Create comparison plots across different configurations
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_ogb_layer_comparison(results_dir, dataset='arxiv', num_layers=8, model='GCN_res'):
    """
    Plot layer-wise sampling comparison (Figure 5/7 style).
    
    Args:
        results_dir: Directory containing .npy result files
        dataset: Dataset name ('arxiv' or 'products')
        num_layers: Number of GCN layers
        model: Model name ('GCN_res' or 'GCN_full')
    """
    p_values = np.arange(0.1, 1.1, 0.1)
    
    # Load results for varying p1 (shallow), fixed p2=0.1
    mean_vary_p1 = []
    for p1 in p_values:
        try:
            filename = f'{num_layers}.{model}.p1={p1:.1f}_p2=0.1_runs=10.npy'
            data = np.load(os.path.join(results_dir, filename), allow_pickle=True)
            mean_acc = np.mean([np.mean(run[-100:]) for run in data])
            mean_vary_p1.append(mean_acc)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            mean_vary_p1.append(np.nan)
    
    # Load results for varying p2 (deep), fixed p1=0.1
    mean_vary_p2 = []
    for p2 in p_values:
        try:
            filename = f'{num_layers}.{model}.p1=0.1_p2={p2:.1f}_runs=10.npy'
            data = np.load(os.path.join(results_dir, filename), allow_pickle=True)
            mean_acc = np.mean([np.mean(run[-100:]) for run in data])
            mean_vary_p2.append(mean_acc)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            mean_vary_p2.append(np.nan)
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    plt.plot(p_values, mean_vary_p1, 'o-', linewidth=3, markersize=10,
             label='Vary p₁ (shallow), p₂=0.1', color='#1f77b4')
    plt.plot(p_values, mean_vary_p2, 's--', linewidth=3, markersize=10,
             label='Vary p₂ (deep), p₁=0.1', color='#ff7f0e')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Sampling Rate', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.title(f'Layer-wise Sampling Effect on {dataset.upper()}', fontsize=16)
    plt.legend(fontsize=14, loc='lower right')
    plt.xticks(p_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    output_file = f'{dataset}_layer_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_2d_heatmap(results_dir, dataset='arxiv', num_layers=8, model='GCN_res'):
    """
    Plot 2D heatmap of test error rate (Figure 6 style).
    
    Args:
        results_dir: Directory containing .npy result files
        dataset: Dataset name
        num_layers: Number of GCN layers
        model: Model name
    """
    p_values = np.arange(0.1, 1.1, 0.1)
    n = len(p_values)
    
    # Load all results into 2D matrix
    matrix = np.zeros((n, n))
    for i, p1 in enumerate(p_values):
        for j, p2 in enumerate(p_values):
            try:
                filename = f'{num_layers}.{model}.p1={p1:.1f}_p2={p2:.1f}_runs=10.npy'
                data = np.load(os.path.join(results_dir, filename), allow_pickle=True)
                mean_acc = np.mean([np.mean(run[-100:]) for run in data])
                matrix[i, j] = 1 - mean_acc  # Convert to error rate
            except FileNotFoundError:
                matrix[i, j] = np.nan
    
    font_size = 16
    
    # Create combined figure
    fig = plt.figure(figsize=(16, 6), dpi=150)
    
    # 2D Heatmap
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(matrix, cmap='inferno_r',
                    extent=[0.1, 1.0, 1.0, 0.1], aspect='auto')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Test Error Rate', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size-2)
    ax1.set_xlabel(r'$p_2$ (deep layer)', fontsize=font_size)
    ax1.set_ylabel(r'$p_1$ (shallow layer)', fontsize=font_size)
    ax1.set_title(f'(a) 2D Heatmap of Test Error Rate', fontsize=font_size)
    ax1.set_xticks(p_values)
    ax1.set_yticks(p_values)
    ax1.tick_params(labelsize=font_size-2)
    ax1.invert_yaxis()
    
    # 3D Surface
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(p_values, p_values)
    surf = ax2.plot_surface(X, Y, matrix, cmap='inferno_r', linewidth=0, antialiased=True)
    ax2.set_xlabel(r'$p_1$', fontsize=font_size, labelpad=10)
    ax2.set_ylabel(r'$p_2$', fontsize=font_size, labelpad=10)
    ax2.set_zlabel('Test Error Rate', fontsize=font_size)
    ax2.set_title(f'(b) 3D Surface of Test Error Rate', fontsize=font_size)
    ax2.view_init(elev=25, azim=45)
    ax2.tick_params(labelsize=font_size-4)
    
    plt.tight_layout()
    
    output_file = f'{dataset}_2d_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_synthetic_results(results_file):
    """
    Plot synthetic experiment results.
    
    Args:
        results_file: Path to .npy file containing results
    """
    results = np.load(results_file, allow_pickle=True).item()
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    markers = ['o', 's', 'D']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for idx, (norm_val, losses) in enumerate(results.items()):
        hidden_channels = [5, 10, 20, 50, 100]
        plt.plot(hidden_channels, np.log10(losses),
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linewidth=3, markersize=10,
                 label=r'$\|A^*\|_1=$' + f'{norm_val:.2f}')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Number of Hidden Neurons', fontsize=16)
    plt.ylabel('Test Error (log₁₀)', fontsize=16)
    plt.title('Test Error vs. Hidden Neurons', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    output_file = 'synthetic_hidden_neurons.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot experiment results')
    parser.add_argument('--type', type=str, required=True,
                        choices=['layer_comparison', '2d_heatmap', 'synthetic'],
                        help='Type of plot to generate')
    parser.add_argument('--results_dir', type=str, default='results/ogb',
                        help='Directory containing result files')
    parser.add_argument('--dataset', type=str, default='arxiv',
                        help='Dataset name')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of GCN layers')
    parser.add_argument('--model', type=str, default='GCN_res',
                        help='Model name')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to results file (for synthetic plots)')
    
    args = parser.parse_args()
    
    if args.type == 'layer_comparison':
        plot_ogb_layer_comparison(args.results_dir, args.dataset, 
                                  args.num_layers, args.model)
    elif args.type == '2d_heatmap':
        plot_2d_heatmap(args.results_dir, args.dataset,
                        args.num_layers, args.model)
    elif args.type == 'synthetic':
        if args.results_file is None:
            raise ValueError("--results_file required for synthetic plots")
        plot_synthetic_results(args.results_file)
