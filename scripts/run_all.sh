#!/bin/bash
# Run all experiments for the paper
# Usage: bash scripts/run_all.sh

set -e

echo "========================================"
echo "GCN Sparsification Experiments"
echo "========================================"

# Create results directories
mkdir -p results/synthetic results/ogb

# =============================================
# Section 4.1: Synthetic Data Experiments
# =============================================

echo ""
echo "Running synthetic experiments..."
echo "----------------------------------------"

# Figure 3: Test error vs hidden neurons
echo "Running: Hidden neurons experiment"
python experiments/synthetic/exp_hidden_neurons.py \
    --epochs 200 \
    --num_runs 3 \
    --save_dir results/synthetic

# Figure 4: 1D sampling rate experiment
echo "Running: 1D sampling rate experiment"
python experiments/synthetic/exp_sampling_rate.py \
    --mode 1d \
    --p1 0.3 \
    --p2 0.3 \
    --epochs 3001 \
    --num_runs 5 \
    --save_dir results/synthetic

# 2D sampling rate experiment (optional, takes longer)
# echo "Running: 2D sampling rate experiment"
# python experiments/synthetic/exp_sampling_rate.py \
#     --mode 2d \
#     --epochs 3001 \
#     --num_runs 3 \
#     --save_dir results/synthetic

# =============================================
# Section 4.3: OGB Dataset Experiments
# =============================================

echo ""
echo "Running OGB experiments..."
echo "----------------------------------------"

# Figure 5: Ogbn-Arxiv - Vary shallow layer sampling
echo "Running: Ogbn-Arxiv (vary p1)"
python experiments/ogb/train_ogbn.py \
    --dataset arxiv \
    --model gcn_res \
    --num_layers 8 \
    --p1_range 0.1 1.0 \
    --p2 0.1 \
    --epochs 800 \
    --runs 10 \
    --save_dir results/ogb

# Figure 5: Ogbn-Arxiv - Vary deep layer sampling
echo "Running: Ogbn-Arxiv (vary p2)"
python experiments/ogb/train_ogbn.py \
    --dataset arxiv \
    --model gcn_res \
    --num_layers 8 \
    --p1 0.1 \
    --p2_range 0.1 1.0 \
    --epochs 800 \
    --runs 10 \
    --save_dir results/ogb

# Figure 6: 2D heatmap (optional, takes much longer)
# echo "Running: Ogbn-Arxiv 2D experiment"
# python experiments/ogb/train_ogbn.py \
#     --dataset arxiv \
#     --model gcn_res \
#     --num_layers 8 \
#     --mode 2d \
#     --epochs 800 \
#     --runs 10 \
#     --save_dir results/ogb

# Figure 7: Ogbn-Products (optional)
# echo "Running: Ogbn-Products"
# python experiments/ogb/train_ogbn.py \
#     --dataset products \
#     --model gcn_res \
#     --num_layers 8 \
#     --p1_range 0.1 1.0 \
#     --p2 0.1 \
#     --epochs 800 \
#     --runs 10 \
#     --save_dir results/ogb

# =============================================
# Generate Plots
# =============================================

echo ""
echo "Generating plots..."
echo "----------------------------------------"

# Plot layer comparison
python scripts/plot_results.py \
    --type layer_comparison \
    --results_dir results/ogb \
    --dataset arxiv \
    --num_layers 8 \
    --model GCN_res

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved in results/"
echo "========================================"
