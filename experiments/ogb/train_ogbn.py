"""
Experiment: Layer-wise sampling on Open Graph Benchmark datasets.

This script evaluates GCN with jumping connections on Ogbn-Arxiv and 
Ogbn-Products datasets, demonstrating that shallow layer sampling has
greater impact on generalization than deep layer sampling.

Usage:
    # Vary p1 (shallow layer), fix p2
    python train_ogbn.py --dataset arxiv --p1_range 0.1 1.0 --p2 0.1
    
    # Vary p2 (deep layer), fix p1
    python train_ogbn.py --dataset arxiv --p1 0.1 --p2_range 0.1 1.0
    
    # 2D grid experiment
    python train_ogbn.py --dataset arxiv --mode 2d
"""

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from src.models import GCNNet, GCN_res


class Logger:
    """Logger for tracking experiment results across runs."""
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        self.results[run].append(result)

    def print_statistics(self):
        results = np.array(self.results)
        best_results = []
        
        for r in results:
            train_acc = r[:, 0].max()
            valid_acc = r[:, 1].max()
            test_idx = r[:, 1].argmax()
            test_acc = r[test_idx, 2]
            best_results.append((train_acc, valid_acc, test_acc))
        
        best_results = np.array(best_results)
        
        print(f'All runs:')
        print(f'  Train: {100 * best_results[:, 0].mean():.2f} ± {100 * best_results[:, 0].std():.2f}')
        print(f'  Valid: {100 * best_results[:, 1].mean():.2f} ± {100 * best_results[:, 1].std():.2f}')
        print(f'  Test:  {100 * best_results[:, 2].mean():.2f} ± {100 * best_results[:, 2].std():.2f}')
        
        return best_results


def sampling(edge_index, p_sampling, p_random=0.1):
    """
    Sample edges by keeping top-k normalized edges with randomization.
    
    Args:
        edge_index: Edge indices
        p_sampling: Fraction of edges to keep
        p_random: Fraction of randomness in selection
    
    Returns:
        Sampled edge_index
    """
    norm_edge, norm_value = gcn_norm(edge_index, add_self_loops=False)
    num_values_to_keep = int(norm_value.numel() * p_sampling)
    num_values_to_delete = norm_value.numel() - num_values_to_keep
    
    top_values, top_indices = torch.topk(norm_value, k=num_values_to_keep)
    other_indices = torch.nonzero(torch.lt(norm_value, top_values[-1])).squeeze()
    
    if num_values_to_keep >= num_values_to_delete:
        num_values_to_random = int(num_values_to_delete * p_random) + 1
    else:
        num_values_to_random = int(num_values_to_keep * p_random) + 1
    
    random_indices = torch.randperm(len(top_indices))[:-num_values_to_random]
    top_indices = top_indices[random_indices]
    
    if len(other_indices.shape) == 0:
        other_indices = other_indices.unsqueeze(0)
    random_indices = torch.randperm(len(other_indices))[:num_values_to_random]
    other_indices = other_indices[random_indices]
    
    choose_indices = torch.cat((top_indices, other_indices))
    sampled_edge_index = norm_edge[:, choose_indices]
    return sampled_edge_index


def load_dataset(dataset_name):
    """Load OGB dataset."""
    if dataset_name == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
        evaluator = Evaluator(name='ogbn-arxiv')
    elif dataset_name == 'products':
        dataset = PygNodePropPredDataset(name='ogbn-products', root='./data/')
        evaluator = Evaluator(name='ogbn-products')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset, evaluator


def train(model, data, train_idx, optimizer, criterion):
    """Training step."""
    model.train()
    out = model(data.x, data.sample1_adj, data.sample2_adj)
    loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    """Evaluation step."""
    model.eval()
    out = model(data.x, data.sample1_adj, data.sample2_adj)
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    
    return train_acc, valid_acc, test_acc


def run_experiment(args):
    """Run the main experiment."""
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset, evaluator = load_dataset(args.dataset)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    data = data.to(device)
    data.edge_index = to_undirected(data.edge_index)
    total_edge_index = data.edge_index
    train_idx = split_idx['train'].to(device)
    
    # Create model
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    
    if args.model == 'gcn_res':
        model = GCN_res(num_features, num_classes, hidden=args.hidden, num_layers=args.num_layers)
    else:
        model = GCNNet(num_features, num_classes, hidden=args.hidden, num_layers=args.num_layers)
    
    model = model.to(device)
    print(f"Model: {model.name}, Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss and optimizer
    criterion = nn.NLLLoss().to(device)
    
    # Prepare sampling rates
    if args.mode == '1d':
        if args.p1_range is not None:
            p1_values = np.arange(args.p1_range[0], args.p1_range[1] + 0.01, 0.1)
            p2_values = [args.p2]
        else:
            p1_values = [args.p1]
            p2_values = np.arange(args.p2_range[0], args.p2_range[1] + 0.01, 0.1)
    else:  # 2d mode
        p1_values = np.arange(0.1, 1.01, 0.1)
        p2_values = np.arange(0.1, 1.01, 0.1)
    
    # Store results
    all_results = {}
    
    for p1 in p1_values:
        for p2 in p2_values:
            p1, p2 = round(p1, 1), round(p2, 1)
            print(f"\n{'='*50}")
            print(f"Running p1={p1:.1f}, p2={p2:.1f}")
            print(f"{'='*50}")
            
            logger = Logger(args.runs)
            run_test_accs = [[] for _ in range(args.runs)]
            
            start_time = time.time()
            
            for run in range(args.runs):
                model.reset_parameters()
                
                # Sample edges for each layer group
                data.edge_index = sampling(total_edge_index, p1, args.p_random)
                data.sample1_adj = T.ToSparseTensor()(data).adj_t
                
                data.edge_index = sampling(total_edge_index, p2, args.p_random)
                data.sample2_adj = T.ToSparseTensor()(data).adj_t
                
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                
                for epoch in range(args.epochs):
                    loss = train(model, data, train_idx, optimizer, criterion)
                    
                    if epoch % 500 == 0 and epoch > 0:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.lr * 0.1
                    
                    result = test(model, data, split_idx, evaluator)
                    train_acc, valid_acc, test_acc = result
                    run_test_accs[run].append(test_acc)
                    
                    if epoch % 100 == 0:
                        print(f'Run: {run + 1:02d}, Epoch: {epoch:03d}, '
                              f'Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}%, Test: {100 * test_acc:.2f}%')
                    
                    logger.add_result(run, result)
            
            best_results = logger.print_statistics()
            
            end_time = time.time()
            print(f"Time: {end_time - start_time:.1f}s")
            
            # Save results
            key = (p1, p2)
            all_results[key] = {
                'test_accs': run_test_accs,
                'best_results': best_results,
                'mean_test': best_results[:, 2].mean(),
                'std_test': best_results[:, 2].std(),
            }
            
            # Save intermediate results
            os.makedirs(args.save_dir, exist_ok=True)
            filename = f'{args.num_layers}.{model.name}.p1={p1:.1f}_p2={p2:.1f}_runs={args.runs}.npy'
            np.save(os.path.join(args.save_dir, filename), run_test_accs)
    
    # Plot results
    if args.mode == '1d':
        plot_1d_results(all_results, p1_values, p2_values, args)
    else:
        plot_2d_results(all_results, p1_values, p2_values, args)
    
    return all_results


def plot_1d_results(all_results, p1_values, p2_values, args):
    """Plot 1D results."""
    plt.figure(figsize=(10, 6), dpi=150)
    
    if len(p2_values) == 1:
        # Varying p1
        p2 = p2_values[0]
        test_accs = [all_results[(p1, p2)]['mean_test'] for p1 in p1_values]
        test_stds = [all_results[(p1, p2)]['std_test'] for p1 in p1_values]
        
        plt.errorbar(p1_values, test_accs, yerr=test_stds, 
                     fmt='o-', linewidth=3, markersize=8, capsize=5,
                     label=f'Vary p₁ (shallow), p₂={p2:.1f}')
        plt.xlabel('p₁ (Shallow Layer Sampling Rate)', fontsize=14)
    else:
        # Varying p2
        p1 = p1_values[0]
        test_accs = [all_results[(p1, p2)]['mean_test'] for p2 in p2_values]
        test_stds = [all_results[(p1, p2)]['std_test'] for p2 in p2_values]
        
        plt.errorbar(p2_values, test_accs, yerr=test_stds,
                     fmt='s--', linewidth=3, markersize=8, capsize=5,
                     label=f'Vary p₂ (deep), p₁={p1:.1f}')
        plt.xlabel('p₂ (Deep Layer Sampling Rate)', fontsize=14)
    
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.title(f'Layer-wise Sampling Effect on {args.dataset.upper()}', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    filename = f'{args.dataset}_sampling_1d.png'
    plt.savefig(os.path.join(args.save_dir, filename), dpi=300)
    plt.close()
    print(f"Plot saved to {args.save_dir}/{filename}")


def plot_2d_results(all_results, p1_values, p2_values, args):
    """Plot 2D heatmap results."""
    # Build 2D matrix
    matrix = np.zeros((len(p1_values), len(p2_values)))
    for i, p1 in enumerate(p1_values):
        for j, p2 in enumerate(p2_values):
            p1, p2 = round(p1, 1), round(p2, 1)
            matrix[i, j] = 1 - all_results[(p1, p2)]['mean_test']  # Error rate
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    font_size = 14
    
    # 2D Heatmap
    im = axes[0].imshow(matrix, cmap='inferno_r',
                        extent=[min(p1_values), max(p1_values), 
                               max(p2_values), min(p2_values)],
                        aspect='auto')
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Test Error Rate', fontsize=font_size)
    axes[0].set_xlabel('p₂', fontsize=font_size)
    axes[0].set_ylabel('p₁', fontsize=font_size)
    axes[0].set_title(f'2D Heatmap ({args.dataset.upper()})', fontsize=font_size)
    axes[0].invert_yaxis()
    
    # 1D comparison: fix one, vary other
    axes[1].plot(p1_values, [all_results[(p1, 0.1)]['mean_test'] for p1 in p1_values],
                 'o-', linewidth=3, markersize=8, label='Vary p₁, p₂=0.1')
    axes[1].plot(p2_values, [all_results[(0.1, p2)]['mean_test'] for p2 in p2_values],
                 's--', linewidth=3, markersize=8, label='Vary p₂, p₁=0.1')
    axes[1].set_xlabel('Sampling Rate', fontsize=font_size)
    axes[1].set_ylabel('Test Accuracy', fontsize=font_size)
    axes[1].set_title('Layer Sensitivity Comparison', fontsize=font_size)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    filename = f'{args.dataset}_sampling_2d.png'
    plt.savefig(os.path.join(args.save_dir, filename), dpi=300)
    plt.close()
    print(f"Plot saved to {args.save_dir}/{filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGB Experiments')
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='arxiv', 
                        choices=['arxiv', 'products'])
    parser.add_argument('--model', type=str, default='gcn_res',
                        choices=['gcn_res', 'gcn_full'])
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=128)
    
    # Sampling parameters
    parser.add_argument('--mode', type=str, default='1d', choices=['1d', '2d'])
    parser.add_argument('--p1', type=float, default=0.1, help='Fixed p1 value')
    parser.add_argument('--p2', type=float, default=0.1, help='Fixed p2 value')
    parser.add_argument('--p1_range', type=float, nargs=2, default=None,
                        help='Range for p1 (start end)')
    parser.add_argument('--p2_range', type=float, nargs=2, default=None,
                        help='Range for p2 (start end)')
    parser.add_argument('--p_random', type=float, default=0.1,
                        help='Randomization factor in sampling')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Output
    parser.add_argument('--save_dir', type=str, default='results/ogb')
    
    args = parser.parse_args()
    run_experiment(args)
