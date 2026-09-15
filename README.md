# Phy-DSTGNN: Physics-Informed Dynamic Spatio-Temporal Graph Neural Network
### Real-Time Urban Drainage Overflow Prediction — Greater Hyderabad (2010–2025)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Research Benchmark](https://img.shields.io/badge/Status-Research%20Benchmark-blue)](https://github.com/Deekshitha-y/phy-dstgnn-urban-drainage)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Case Study](https://img.shields.io/badge/Case%20Study-Hyderabad%20GHMC%202010--2025-brightgreen)](https://www.ghmc.gov.in/)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

---

## 📖 Overview

**Phy-DSTGNN** is a novel deep learning framework for real-time urban flood prediction and drainage surcharge forecasting across the **72-manhole drainage network of Greater Hyderabad (GHMC)**. It combines:

- 🧠 **Physics-Informed Learning** — Saint-Venant 1D equations embedded in the training loss
- 🕸️ **Dynamic Directed Graph Attention (GATv2)** — with residual skip-connections over the drainage topology
- ⏱️ **Gated Dilated Temporal Convolution (TCN)** — for multi-scale temporal patterns
- 🔍 **Multi-Head Self-Attention** — cross-node interaction at each forecast horizon
- 🎯 **Monte Carlo Dropout** — for 95% confidence uncertainty quantification
- 🔬 **Integrated Gradients (XAI)** — axiomatic path-Shapley feature attribution
- 🗺️ **Live GIS Dashboard** — 6-zone real-time risk map with GHMC disaster protocols

Trained on **15 years of monsoon climate data (2010–2025)** comprising 140,256 hourly records and the historic October 2020 super-flood benchmark.

---

## 🏆 Key Results

### Model Performance vs. Baseline LSTM

| Metric | Baseline LSTM | **Phy-DSTGNN (Ours)** | Improvement |
|:---|:---:|:---:|:---:|
| **Nash-Sutcliffe Efficiency (NSE)** | 0.8809 | **0.8949** | +0.014 |
| **Pearson R²** | 0.8824 | **0.8949** | +0.0125 |
| **RMSE (Normalized Depth)** | 0.3440 | **0.3196** | −7.1% |
| **Peak Flood Depth Error** | 25.99% | **4.64%** | **−82.1%** ⭐ |
| **Flood Recall / POD** | 59.52% | **76.27%** | **+16.75%** ⭐ |
| **Critical Success Index (CSI)** | 0.5349 | **0.6215** | +0.0866 |
| **F1-Score** | 0.6970 | **0.7666** | +0.0696 |
| **ROC-AUC** | 0.997 | **0.998** | +0.001 |

### Multi-Horizon NSE (Phy-DSTGNN)

| Horizon | NSE | RMSE | Operational Use |
|:---|:---:|:---:|:---|
| **t+5 min** | 0.9787 | 0.1439 | Pump trigger signals |
| **t+15 min** | 0.9714 | 0.1666 | Pre-emptive gate control |
| **t+30 min** | 0.9524 | 0.2150 | Sluice gate operation |
| **t+60 min** | 0.8963 | 0.3176 | Traffic diversion alerts |
| **t+120 min** | 0.6762 | 0.5615 | Evacuation planning |

### XAI — Integrated Gradients Feature Attribution

| Rank | Feature | Attribution |
|:---|:---|:---:|
| 1 | Water Depth & Hydraulic Surcharge | **37.3%** |
| 2 | Subcatchment Area & Runoff Volume | **17.2%** |
| 3 | Impervious Surface Coverage | **15.7%** |
| 4 | Pipe Invert Elevation & Bed Slope | **13.6%** |
| 5 | Antecedent Rainfall Intensity | **9.0%** |
| 6 | Ground Surface Elevation (DEM) | **7.2%** |

---

## 🏗️ Architecture

```
Input Features (6D × 72 nodes × T timesteps)
         │
         ▼
┌─────────────────────────────────────┐
│   Residual Directed GATv2           │
│   (hidden=96, heads=4, skip-proj)   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Gated Dilated TCN                 │
│   (kernel=3, dilation=2, dropout)   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Multi-Head Self-Attention         │
│   (4 heads, d_model=96)             │
└────────────────┬────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  Regression Head    Classification Head
  (depth forecast)   (overflow sigmoid)
  5 × 72 nodes       5 × 72 nodes
```

**Loss Function:**
```
L = SmoothL1(depth) + λ_focal × FocalLoss(α=0.65, γ=1.5)
  + λ_mass × MassConservation + λ_peak × AsymmetricPeakPenalty
```
where `λ_focal=0.15`, `λ_mass=0.02`, `λ_peak=0.15`

**Parameters:**
- Phy-DSTGNN: **103,306 parameters**
- Baseline LSTM: 11,242 parameters
- Training: AdamW (lr=2e-3) + CosineAnnealingLR + EarlyStopping (patience=6)

---

## 📁 Repository Structure

```
phy-dstgnn-urban-drainage/
│
├── major_project_3.ipynb         # Complete 12-cell research notebook ⭐
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ARCHITECTURE.md               # Detailed math & tensor dimensions
├── DATASET_GUIDE.md              # Data sources & preprocessing guide
├── PROJECT_PLAN.md               # 6-phase research roadmap
│
├── data/
│   ├── hyderabad_gis/
│   │   └── hyderabad_drainage_network.geojson   # 72-node drainage topology
│   ├── processed/
│   │   ├── nodes.csv                             # Node features (72 manholes)
│   │   ├── edges.csv                             # Pipe connectivity (66 edges)
│   │   ├── landcover_15year_timeseries.csv       # LULC time series
│   │   └── topography_summary.json
│   ├── rainfall_raw/
│   │   └── hyderabad_hourly_rainfall_2010_2025.csv  # 140,256 hourly records
│   └── synthetic_storms/
│       └── *.csv                                 # Chicago design storms (Tr: 2–100 yr)
│
├── src/
│   └── data/
│       ├── rainfall_engine.py                    # Climate data generation
│       ├── drainage_topology_extractor.py        # Graph construction
│       ├── hyetograph_generator.py               # Storm profile generation
│       ├── elevation_topography.py               # DEM processing
│       └── landcover_processor.py                # LULC classification
│
├── tests/
│   └── test_phase1.py                            # Phase 1 pipeline tests
│
└── references/                                   # 15 IEEE/Nature reference PDFs
    ├── Vaswani_et_al_Attention_Is_All_You_Need_2017.pdf
    ├── Raissi_et_al_PINN_2019.pdf
    ├── Brody_et_al_GATv2_2022.pdf
    └── ...
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python >= 3.10
CUDA-compatible GPU (recommended: Tesla T4 or higher)
```

### 1. Clone the repository
```bash
git clone https://github.com/Deekshitha-y/phy-dstgnn-urban-drainage.git
cd phy-dstgnn-urban-drainage
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `major_project_3.ipynb` in **Google Colab** (recommended for GPU access) or Jupyter:

```bash
jupyter notebook major_project_3.ipynb
```

> **Recommended:** Run on [Google Colab](https://colab.research.google.com/) with **Tesla T4 GPU** runtime for full training (~8–12 minutes).

### 4. Cell-by-Cell Execution Guide

| Cell | Description | Runtime |
|:---:|:---|:---:|
| **Cell 1** | Environment setup, GPU check, seed fixation | ~30s |
| **Cell 2** | 15-year climate dataset generation (140,256 hrs) | ~45s |
| **Cell 3** | 72-node drainage graph & 1D hydrodynamic simulation | ~60s |
| **Cell 4** | PyTorch DataLoaders (zero data leakage, 70/15/15 split) | ~20s |
| **Cell 5** | Phy-DSTGNN + Baseline LSTM model definitions | ~10s |
| **Cell 6** | Training loop (AdamW + CosineAnnealingLR + EarlyStopping) | ~8 min |
| **Cell 7** | Comprehensive benchmark evaluation — 8-metric comparison table | ~30s |
| **Cell 8** | XAI (Integrated Gradients) + MC Dropout UQ + Hydrograph | ~45s |
| **Cell 9** | Live interactive GIS dashboard (6-zone risk map) | ~30s |
| **Cell 10** | 9 Publication-quality research figures (300 DPI PNG + PDF) | ~60s |
| **Cell 11** | Model checkpoint save + complete session summary | ~10s |
| **Cell 12** | Auto-download all outputs to local PC | ~10s |

---

## 📊 Dataset

### Overview
- **Source:** 1D Hydrodynamic Simulation (Saint-Venant equations + Manning's routing)
- **Duration:** 15 years (January 2010 — December 2025)
- **Resolution:** Hourly (140,256 records)
- **Coverage:** Greater Hyderabad — 6 urban corridors, 72 drainage nodes

### Study Area Zones

| Zone | Area (km²) | Risk Level | Key Characteristic |
|:---|:---:|:---:|:---|
| **Begumpet** | 18.4 | 🔴 HIGH | Low-lying, high imperviousness |
| **L.B. Nagar** | 22.1 | 🟠 MEDIUM-HIGH | Dense urban, poor gradient |
| **Kukatpally** | 31.2 | 🟡 MEDIUM | Mixed residential-commercial |
| **Secunderabad** | 15.8 | 🟢 MODERATE | Better drainage infrastructure |
| **Madhapur** | 28.6 | 🟢 MODERATE | IT corridor, newer drainage |
| **Gachibowli** | 35.3 | 🔵 LOW | Modern infrastructure, good slope |

### Key Statistics
- 📊 Total hourly records: **140,256**
- 🌧️ Monsoon sequences: **58,752 hrs** (June–October)
- 💧 Overflow events: **7,150 surcharge instances** (0.83% frequency)
- ⛈️ October 2020 super-flood: **22.33 mm/hr peak intensity**
- 🌧️ Annual average rainfall: **1,588.5 mm/year**

---

## 🔬 Explainability & Uncertainty

### Integrated Gradients (Path-Shapley Attribution)
Axiomatic attribution from baseline (zero tensor) to each input feature via 50-step Riemann integration. Identifies **Water Depth (37.3%)** as the dominant driver of overflow prediction.

### Monte Carlo Dropout (UQ)
- **30 stochastic forward passes** with dropout enabled at inference
- **Mean overflow probability:** 1.0%
- **Average confidence score:** 98.3%
- **95% CI width:** ±0.0498 (normalized depth units)

---

## 🗺️ Real-Time GIS Dashboard

The interactive dashboard (Cell 9) provides:
- **4-column KPI cards:** Peak Risk %, Confidence Score, Risk Level, Peak Water Depth
- **Recommended Action Protocol** (GHMC Disaster Management procedures)
- **Knowledge Graph** reasoning chain: Rain → Runoff → Surcharge → Overflow
- **SHAP attribution bars** for the top 6 features
- **Folium map** with colour-coded risk nodes across 72 manholes

Risk tiers:
- 🔴 **HIGH** (≥70%) — Immediate evacuation protocol
- 🟠 **MEDIUM** (≥48%) — Pump activation + traffic diversion
- 🟡 **MODERATE** (≥25%) — Monitoring + pre-emptive alerts
- 🔵 **LOW** (<25%) — Normal operations

---

## 📄 Research & Evaluation Figures

Running Cells 10 & 10B generates **9 publication-quality research figures** at 300 DPI:

| Figure | Description |
|:---|:---|
| **(a)** | Multi-Task Physics Loss Convergence (both models) |
| **(b)** | Multi-Horizon Lead-Time NSE Degradation |
| **(c)** | Surcharge Overflow Precision-Recall Curve |
| **(d)** | Scatter Parity Plot (Pearson R² = 0.8949) |
| **(e)** | Integrated Gradients Feature Attribution (Path-Shapley) |
| **(f)** | Extreme Flood Peak Depth Error Comparison |
| **(g)** | ROC-AUC Curve (AUC = 0.998 vs 0.997) |
| **(h)** | Prediction Residual Error Distribution |
| **(i)** | Spatial Node-Wise RMSE Heatmap (72 manholes) |

---

## 📦 Output Files

After running all 12 cells, the following files are generated:

| File | Description |
|:---|:---|
| `phy_dstgnn_final.pth` | Trained Phy-DSTGNN model weights (103,306 params) |
| `baseline_lstm_final.pth` | Trained Baseline LSTM weights (11,242 params) |
| `IEEE_Research_Figures_Full.png` | 6 main research figures (300 DPI) |
| `IEEE_Research_Figures_Full.pdf` | Vector PDF for publication/presentation |
| `IEEE_Supplementary_Figures.png` | 3 supplementary figures (300 DPI) |
| `IEEE_Supplementary_Figures.pdf` | Supplementary vector PDF |
| `PhyDSTGNN_GreaterHyderabad_Complete_Output.zip` | All of the above bundled |

---

## 🛠️ Technology Stack

| Category | Libraries |
|:---|:---|
| **Deep Learning** | `PyTorch 2.x`, `torch_geometric` |
| **Scientific Computing** | `NumPy`, `SciPy`, `scikit-learn` |
| **Geospatial** | `GeoPandas`, `Folium`, `Shapely`, `OSMnx` |
| **Visualization** | `Matplotlib`, `Seaborn`, `Plotly` |
| **Data Handling** | `Pandas`, `xarray`, `h5py` |
| **XAI** | Integrated Gradients (custom, axiomatic) |
| **UQ** | Monte Carlo Dropout (custom, 30 passes) |
| **Dashboard** | `ipywidgets`, `IPython.display`, `Folium` |

---

## 📚 References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
2. Raissi et al., *Physics-Informed Neural Networks*, JCP 2019
3. Brody et al., *How Attentive are Graph Attention Networks? (GATv2)*, ICLR 2022
4. Bai et al., *TCN for Sequence Modeling*, ICLR 2018
5. Gal & Ghahramani, *Dropout as Bayesian Approximation*, ICML 2016
6. Lundberg & Lee, *SHAP — A Unified Approach to Feature Attribution*, NeurIPS 2017
7. Sundararajan et al., *Integrated Gradients*, ICML 2017
8. Wu et al., *Graph WaveNet*, IJCAI 2019
9. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
10. Nearing et al., *Global Flood Forecasting*, Nature 2024

Full PDFs available in the [`references/`](references/) folder.

---

## 👩‍💻 Author

**Deekshitha Y**
B.Tech — Computer Science Engineering
Project: Major Project — Research & Engineering Implementation

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for real-time urban flood resilience in Greater Hyderabad.*
