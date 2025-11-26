"""
Utility functions for GCN experiments.

This module contains:
- Graph generation utilities
- Adjacency matrix normalization
- Data splitting
- Sampling functions
- Target function generation
"""

import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def generate_edge_index(N=2000, min_degree=1, max_degree=200, mean_degree=200, std_degree=20):
    """
    Generate edge index with node degrees following a truncated normal distribution.
    
    Args:
        N: Number of nodes
        min_degree: Minimum node degree
        max_degree: Maximum node degree
        mean_degree: Mean of degree distribution
        std_degree: Standard deviation of degree distribution
    
    Returns:
        edge_index: [2, num_edges] tensor
    """
    # Generate normal random degrees
    degrees = np.random.normal(loc=mean_degree, scale=std_degree, size=N).astype(int)
    degrees = np.clip(degrees, min_degree, max_degree)

    rows = []
    cols = []

    for node, degree in enumerate(degrees):
        neighbors = np.random.choice(np.delete(np.arange(N), node), degree, replace=False)
        rows.extend([node] * degree)
        cols.extend(neighbors)

    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    return edge_index


def generate_sbm_edge_index(N, p1, p2, c=100, sigma=0.005):
    """
    Generate edge index using Stochastic Block Model (SBM).
    
    Creates a graph with two communities where:
    - Within-community edge probability is higher
    - Between-community edge probability is lower
    
    Args:
        N: Number of nodes
        p1: Base edge probability
        p2: Modified edge probability for cross-community edges
        c: Boundary between communities (nodes 0 to c-1 vs c to N-1)
        sigma: Noise standard deviation
    
    Returns:
        edge_index: [2, num_edges] tensor
    """
    noise = torch.normal(mean=0, std=sigma, size=(N, N))

    # Create probability matrix
    prob_matrix = torch.full((N, N), p1)
    prob_matrix[c:, :] += (p2 - p1)
    prob_matrix[:, c:] += (p2 - p1)
    prob_matrix += noise
    prob_matrix = torch.clamp(prob_matrix, 0, 1)

    # Generate symmetric adjacency matrix
    mask = torch.triu(torch.ones(N, N), diagonal=1)
    adj_matrix = torch.bernoulli(prob_matrix) * mask
    symmetric_adj_matrix = adj_matrix + adj_matrix.T.clone()

    edge_index = torch.nonzero(symmetric_adj_matrix, as_tuple=False).t()
    return edge_index


def generate_normalized_adj_matrix(edge_index, num_nodes):
    """
    Generate normalized adjacency matrix: D^{-1/2} A D^{-1/2}.
    
    Args:
        edge_index: [2, num_edges] tensor
        num_nodes: Number of nodes
    
    Returns:
        normalized_adj_matrix: [N, N] dense tensor
    """
    # Create adjacency matrix
    adj_matrix = torch.sparse_coo_tensor(
        edge_index, 
        torch.ones(edge_index.size(1)), 
        torch.Size([num_nodes, num_nodes])
    )
    adj_matrix = adj_matrix.to_dense()

    # Add self-loops
    adj_matrix += torch.eye(num_nodes)

    # Compute D^{-1/2}
    degree = torch.sum(adj_matrix, dim=1)
    D_inv_sqrt = torch.diag(1 / torch.sqrt(degree))

    # Normalized adjacency
    normalized_adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return normalized_adj_matrix


def generate_target_data(A_star, X, W, V, C, alpha):
    """
    Generate node labels using the target function from the paper.
    
    Target function: H = F(A*, X) + α * G(F(A*, X))
    where:
        F(A*, X) = A* X W C
        G(F) = sin(A* F V) * tanh(A* F V) @ C
    
    Args:
        A_star: Normalized adjacency matrix [N, N]
        X: Node features [N, d]
        W: Weight matrix [d, m]
        V: Weight matrix [k, m]
        C: Weight matrix [m, k]
        alpha: Coefficient for the composite function G
    
    Returns:
        node_labels: [N, k] tensor
    """
    # F(A*, X) = A* X W C
    node_labels_F = torch.mm(A_star, torch.mm(X, W))
    node_labels_F = torch.mm(node_labels_F, C)  # [N, k]
    
    # G(F) = sin(A* F V) * tanh(A* F V) @ C
    inner = torch.mm(A_star, torch.mm(node_labels_F, V))
    node_labels_G_F_sin = torch.sin(inner)
    node_labels_G_F_tanh = torch.tanh(inner)
    node_labels_G_F = torch.mm(node_labels_G_F_sin * node_labels_G_F_tanh, C)  # [N, k]
    
    # H = F + α * G(F)
    node_labels_H = node_labels_F + alpha * node_labels_G_F
    return node_labels_H


def split_masks(num_nodes, train_rate, val_rate, test_rate):
    """
    Split nodes into train/validation/test sets.
    
    Args:
        num_nodes: Total number of nodes
        train_rate: Fraction of nodes for training
        val_rate: Fraction of nodes for validation
        test_rate: Fraction of nodes for testing
    
    Returns:
        train_mask, val_mask, test_mask: Boolean tensors
    """
    assert abs(train_rate + val_rate + test_rate - 1.0) < 1e-6, "Rates must sum to 1"

    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_end = int(num_nodes * train_rate)
    val_end = train_end + int(num_nodes * val_rate)

    masks = np.zeros(num_nodes, dtype=bool)
    
    masks[indices[:train_end]] = True
    train_mask = torch.tensor(masks.copy())

    masks[:] = False
    masks[indices[train_end:val_end]] = True
    val_mask = torch.tensor(masks.copy())

    masks[:] = False
    masks[indices[val_end:]] = True
    test_mask = torch.tensor(masks.copy())

    return train_mask, val_mask, test_mask


def sampling_topk(edge_index, p_sampling):
    """
    Sample edges by keeping top-k edges based on normalized edge weights.
    
    Args:
        edge_index: [2, num_edges] tensor
        p_sampling: Fraction of edges to keep (0, 1]
    
    Returns:
        sampled_edge_index: [2, num_sampled_edges] tensor
    """
    norm_edge, norm_value = gcn_norm(edge_index, add_self_loops=False)
    num_values_to_keep = int(norm_value.numel() * p_sampling)
    _, top_indices = torch.topk(norm_value, k=num_values_to_keep)
    sampled_edge_index = norm_edge[:, top_indices]
    return sampled_edge_index


def sampling_with_random(edge_index, p_sampling, p_random=0.1):
    """
    Sample edges with randomization: keep top-k edges plus random sampling.
    
    This helps avoid overfitting to specific edge patterns.
    
    Args:
        edge_index: [2, num_edges] tensor
        p_sampling: Fraction of edges to keep (0, 1]
        p_random: Fraction of randomness in selection
    
    Returns:
        sampled_edge_index: [2, num_sampled_edges] tensor
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
    
    # Remove some top indices
    random_indices = torch.randperm(len(top_indices))[:-num_values_to_random]
    top_indices = top_indices[random_indices]
    
    # Add some random indices from other edges
    if len(other_indices.shape) == 0:
        other_indices = other_indices.unsqueeze(0)
    random_indices = torch.randperm(len(other_indices))[:num_values_to_random]
    other_indices = other_indices[random_indices]
    
    choose_indices = torch.cat((top_indices, other_indices))
    sampled_edge_index = norm_edge[:, choose_indices]
    return sampled_edge_index


def compute_adjacency_norm(A):
    """
    Compute the 1-norm (maximum column sum) of adjacency matrix.
    
    Args:
        A: Adjacency matrix [N, N]
    
    Returns:
        one_norm: Scalar value
    """
    return torch.norm(A, p=1, dim=0).max()
