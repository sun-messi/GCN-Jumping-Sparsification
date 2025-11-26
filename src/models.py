"""
GCN model definitions with jumping connections.

This module contains:
- GCN: Basic two-layer GCN with skip connection for synthetic experiments
- GCNNet: Multi-layer GCN without skip connection
- GCN_res: Multi-layer GCN with learnable residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Two-layer GCN with skip connection for synthetic experiments.
    
    Architecture:
        x -> Linear -> ReLU -> A @ h1 -> Linear -> ReLU -> A @ h2 -> h1 + h2 -> Linear -> output
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.layer_1 = nn.Linear(in_channels, hidden_channels)
        self.layer_2 = nn.Linear(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, out_channels)
        nn.init.normal_(self.out.weight, mean=0.0, std=1.0)

    def forward(self, x, A1, A2=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            A1: Normalized adjacency matrix for layer 1 [N, N]
            A2: Normalized adjacency matrix for layer 2 [N, N] (optional, defaults to A1)
        
        Returns:
            Output predictions [N, out_channels]
        """
        if A2 is None:
            A2 = A1
        
        hidden_1 = F.relu(A1 @ self.layer_1(x))
        hidden_2 = F.relu(A2 @ self.layer_2(hidden_1))
        # Skip connection
        added_12 = hidden_1 + hidden_2
        logits = self.out(added_12)
        return logits


class GCN_PyG(nn.Module):
    """
    Two-layer GCN using PyTorch Geometric GCNConv.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_PyG, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out_fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        # Skip connection
        x_out = x1 + x2
        x_out = self.out_fc(x_out)
        return x_out


class GCNNet(nn.Module):
    """
    Multi-layer GCN without skip connections.
    
    Args:
        num_features: Input feature dimension
        num_classes: Number of output classes
        hidden: Hidden layer dimension (default: 256)
        num_layers: Number of GCN layers (default: 6)
    """
    def __init__(self, num_features, num_classes, hidden=256, num_layers=6):
        super(GCNNet, self).__init__()
        self.name = 'GCN_full'
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.input_fc = nn.Linear(num_features, hidden)

        for _ in range(self.num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.out_fc = nn.Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, x, sample1_adj, sample2_adj):
        """
        Forward pass with layer-wise sampling.
        
        Args:
            x: Node features
            sample1_adj: Sparse adjacency for shallow layers (first half)
            sample2_adj: Sparse adjacency for deep layers (second half)
        """
        x = self.input_fc(x)

        for i in range(self.num_layers):
            if i < self.num_layers // 2:
                x = self.convs[i](x, sample1_adj)
            else:
                x = self.convs[i](x, sample2_adj)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class GCN_res(nn.Module):
    """
    Multi-layer GCN with learnable residual/jumping connections.
    
    The output is a weighted sum of all layer outputs, where weights are learned.
    
    Args:
        num_features: Input feature dimension
        num_classes: Number of output classes
        hidden: Hidden layer dimension (default: 256)
        num_layers: Number of GCN layers (default: 6)
    """
    def __init__(self, num_features, num_classes, hidden=256, num_layers=6):
        super(GCN_res, self).__init__()
        self.name = 'GCN_res'
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.input_fc = nn.Linear(num_features, hidden)

        for _ in range(self.num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.out_fc = nn.Linear(hidden, num_classes)
        # Learnable weights for combining layer outputs
        self.weights = nn.Parameter(torch.randn(len(self.convs)))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()
        nn.init.normal_(self.weights)

    def forward(self, x, sample1_adj, sample2_adj):
        """
        Forward pass with layer-wise sampling and residual aggregation.
        
        Args:
            x: Node features
            sample1_adj: Sparse adjacency for shallow layers (first half)
            sample2_adj: Sparse adjacency for deep layers (second half)
        """
        x = self.input_fc(x)

        layer_out = []
        for i in range(self.num_layers):
            if i < self.num_layers // 2:
                x = self.convs[i](x, sample1_adj)
            else:
                x = self.convs[i](x, sample2_adj)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.5, training=self.training)
            layer_out.append(x)

        # Weighted sum of all layer outputs
        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]

        x = sum(layer_out)
        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)
        return x
