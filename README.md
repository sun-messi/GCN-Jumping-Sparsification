# Theoretical Learning Performance of Graph Networks: the Impact of Jumping Connections and Layer-wise Sparsification

[![TMLR](https://img.shields.io/badge/TMLR-2025-blue)](https://openreview.net/forum?id=Q9AkJpfJks)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Official implementation for the paper:

> **Theoretical Learning Performance of Graph Networks: the Impact of Jumping Connections and Layer-wise Sparsification**  
> Jiawei Sun, Hongkang Li, Meng Wang  
> *Transactions on Machine Learning Research (TMLR), 2025*

[[Paper]](https://openreview.net/pdf?id=Q9AkJpfJks) [[OpenReview]](https://openreview.net/forum?id=Q9AkJpfJks)

## Overview

This paper presents the first learning dynamics and generalization analysis of GCNs with jumping connections using graph sparsification. Our key findings:

- **Layer-wise sparsification requirements**: Jumping connections lead to different sparsification requirements across layers
- **Shallow layer sensitivity**: In deep GCNs, generalization is more affected by the sparsified matrix deviations in shallow layers than deep layers
- **Practical guideline**: Apply conservative sparsification in shallow layers to preserve local neighborhood information, and more aggressive pruning in deeper layers

<p align="center">
  <img src="figures/overview.png" width="600"/>
</p>

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gcn-sparsification.git
cd gcn-sparsification

# Create conda environment (recommended)
conda create -n gcn-sparse python=3.9
conda activate gcn-sparse

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
gcn-sparsification/
├── src/
│   ├── models.py           # GCN model definitions (GCN, GCN_res)
│   ├── utils.py            # Utility functions (data generation, sampling, etc.)
│   └── gcn_norm.py         # GCN normalization utilities
├── experiments/
│   ├── synthetic/          # Section 4.1: Synthetic data experiments
│   │   ├── exp_hidden_neurons.py    # Test error vs hidden neurons (Figure 3)
│   │   ├── exp_sampling_rate.py     # Sampling rate effects (Figure 4)
│   │   └── exp_2d_sampling.py       # 2D sampling rate grid
│   └── ogb/                # Section 4.3: Large-scale real datasets
│       ├── train_ogbn_arxiv.py      # Ogbn-Arxiv experiments (Figure 5, 6)
│       └── train_ogbn_products.py   # Ogbn-Products experiments (Figure 7)
├── scripts/
│   └── plot_results.py     # Visualization scripts
├── figures/                # Output figures
├── requirements.txt
└── README.md
```

## Experiments

### 4.1 Synthetic Data Experiments

Generate synthetic graphs and evaluate GCN performance under different configurations.

**Effect of Hidden Neurons (Figure 3):**
```bash
python experiments/synthetic/exp_hidden_neurons.py
```

**Layer-wise Sampling Rate (Figure 4):**
```bash
python experiments/synthetic/exp_sampling_rate.py --p1 0.5 --p2 0.5
```

**2D Sampling Rate Grid:**
```bash
python experiments/synthetic/exp_2d_sampling.py
```

### 4.3 Large-scale Real Dataset Experiments

Experiments on Open Graph Benchmark (OGB) datasets.

**Ogbn-Arxiv (Figure 5, 6):**
```bash
# Vary sampling rate in shallow layers (p1), fix deep layers (p2=0.1)
python experiments/ogb/train_ogbn_arxiv.py --p1_range 0.1 1.0 --p2 0.1

# Vary sampling rate in deep layers (p2), fix shallow layers (p1=0.1)  
python experiments/ogb/train_ogbn_arxiv.py --p1 0.1 --p2_range 0.1 1.0

# 2D heatmap experiment
python experiments/ogb/train_ogbn_arxiv.py --mode 2d
```

**Ogbn-Products (Figure 7):**
```bash
python experiments/ogb/train_ogbn_products.py --p1_range 0.1 1.0 --p2 0.1
```

## Key Results

### Layer-wise Sampling Sensitivity

Our experiments demonstrate that **shallow layers are more sensitive to sparsification** than deep layers:

| Dataset | Varying p₁ (shallow) | Varying p₂ (deep) |
|---------|---------------------|-------------------|
| Ogbn-Arxiv | High sensitivity | Low sensitivity |
| Ogbn-Products | High sensitivity | Low sensitivity |

### Practical Recommendations

Based on our theoretical analysis and experiments:

1. **Conservative sparsification in shallow layers**: Preserve local neighborhood information
2. **Aggressive pruning in deeper layers**: Especially when skip connections are present
3. **Retain high-weight edges**: Edges with larger normalized weights contribute more to message propagation

## Citation

If you find this work useful, please cite:

```bibtex
@article{suntheoretical,
  title={Theoretical Learning Performance of Graph Networks: the Impact of Jumping Connections and Layer-wise Sparsification},
  author={Sun, Jiawei and Li, Hongkang and Wang, Meng},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Open Graph Benchmark (OGB) team for providing the benchmark datasets
- PyTorch Geometric team for the excellent graph learning library
