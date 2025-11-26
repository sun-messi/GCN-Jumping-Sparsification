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

### 4.3 Large-scale Real Dataset Experiments

Experiments on Open Graph Benchmark (OGB) datasets, specifically **Ogbn-Arxiv**, using an 8-layer Jumping Knowledge Network.

#### 1. Layer-wise Pruning Sensitivity

<p align="center">
<table>
  <tr>
    <td align="center" width="50%">
      <img src="figure/3D_GCN_res.png" alt="3D GCN Results">
      <br><strong>(a) Deeper layers tolerate higher sampling rates</strong>
    </td>
    <td align="center" width="50%">
      <img src="figure/2D_GCN_res.png" alt="2D GCN Results">
      <br><strong>(b) 2D Heatmap of Test Error</strong>
    </td>
  </tr>
</table>
</p>

> **Figure Analysis**: Learning deep GCNs on Ogbn-Arxiv. Deeper layers tolerate higher sampling rates than shallow layers while maintaining accuracy. The test error decreases more drastically when increasing retention in shallow layers ($q_1$) compared to deep layers ($q_2$).

---

#### Edge Weight Influence Analysis

<p align="center">
<table>
  <tr>
    <td align="center" width="50%">
      <img src="figure/3D_Heatmap_of_Test_Error_Rate2.png" alt="3D Heatmap Edge Weight">
      <br><strong>(a) Retaining large-weight edges</strong>
    </td>
    <td align="center" width="50%">
      <img src="figure/2D_Heatmap_of_Test_Error_Rate.png" alt="2D Heatmap Edge Weight">
      <br><strong>(b) 2D Heatmap of Test Error</strong>
    </td>
  </tr>
</table>
</p>

> **Figure Analysis**: Learning deep GCNs on Ogbn-Arxiv. Retaining more large-weight edges (small $s_1, s_2$) outperforms retaining more small-weight edges (large $s_1, s_2$).


## Key Results

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
