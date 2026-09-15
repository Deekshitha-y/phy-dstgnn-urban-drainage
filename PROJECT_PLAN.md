# AI-Based Urban Drainage Overflow Prediction
## High-Impact Project Implementation Plan: Physics-Informed Dynamic Spatio-Temporal Graph Neural Network (Phy-DSTGNN)

---

## 📌 Executive Summary & Problem Statement
* **Project Title:** Physics-Informed Dynamic Spatio-Temporal Graph Neural Network (Phy-DSTGNN) for Real-Time Urban Drainage Overflow & Inundation Forecasting.
* **Target Venue:** Top-tier Artificial Intelligence & Environmental Computing Venues.
* **Study Area:** Greater Hyderabad Municipal Corporation (GHMC), Telangana, India ($17.3850^\circ\text{ N}, 78.4867^\circ\text{ E}$).
* **Temporal Scope:** 15 Years of Multi-Source Meteorological, Geospatial, and Drainage Data (**2010 – 2025**).
* **Core Problem:** Heavy monsoon cloudbursts cause catastrophic urban drainage surcharge and street flooding. Existing AI solutions in literature rely on isolated time-series models (LSTM) or static GNNs that ignore dynamic hydraulic gradients, violate physical mass conservation, and severely underestimate rare flood peaks.

---

## 🎯 Benchmark Performance Targets vs. Existing Literature

| Metric Category | Metric | Traditional Hydrodynamic (EPA-SWMM) | Baseline ML/DL (LSTM, CNN-LSTM) | Baseline GNN (STGCN, DCRNN) | **Proposed Phy-DSTGNN (Our Target)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hydraulic Accuracy** | **Nash-Sutcliffe Efficiency (NSE)** | $1.00$ (Ground Truth) | $0.72 - 0.81$ | $0.84 - 0.89$ | **$\mathbf{\ge 0.96 - 0.98}$** |
| | **$R^2$ Score** | $1.00$ | $0.78 - 0.85$ | $0.88 - 0.92$ | **$\mathbf{\ge 0.97 - 0.99}$** |
| | **RMSE (Water Depth $H$)** | $0.00\text{ m}$ | $0.18 - 0.35\text{ m}$ | $0.09 - 0.15\text{ m}$ | **$\mathbf{\le 0.03 - 0.05\text{ m}}$** |
| | **Peak Depth Error ($\Delta H_{\text{peak}}$)** | Baseline | $18\% - 30\%$ | $9\% - 15\%$ | **$\mathbf{\le 3\% - 5\%}$** |
| | **Peak Timing Error ($\Delta T_{\text{peak}}$)** | Baseline | $\pm 15 - 30\text{ min}$ | $\pm 5 - 15\text{ min}$ | **$\mathbf{\le 0 - 5\text{ min}}$** |
| **Overflow Early Warning** | **Critical Success Index (CSI)** | N/A | $0.55 - 0.65$ | $0.72 - 0.79$ | **$\mathbf{\ge 0.90 - 0.94}$** |
| | **Probability of Detection (POD)** | N/A | $70\% - 78\%$ | $82\% - 88\%$ | **$\mathbf{\ge 95.0\%}$** |
| | **False Alarm Ratio (FAR)** | N/A | $22\% - 35\%$ | $12\% - 18\%$ | **$\mathbf{\le 5.0\%}$** |
| | **F1-Score (Overflow Events)** | N/A | $0.68 - 0.74$ | $0.80 - 0.85$ | **$\mathbf{\ge 0.93 - 0.96}$** |
| **Physics & Speed** | **Mass Balance Violation Rate** | $0\%$ | $25\% - 42\%$ | $14\% - 22\%$ | **$\mathbf{\le 1.5\%}$** |
| | **Inference Time (Full City)** | $\sim 10 - 30\text{ mins}$ | $\sim 1.2\text{ s}$ | $\sim 0.4\text{ s}$ | **$\mathbf{\le 0.035\text{ s (Real-time)}}$** |

---

## 🏛️ End-to-End System Architecture

```
========================================================================================================================
                                    STAGE 1: MULTI-MODAL GEOSPATIAL & METEOROLOGICAL INGESTION
========================================================================================================================
  [Copernicus DEM 30m / CartoDEM]  [ESA WorldCover 10m / Dynamic World]  [OpenStreetMap / GHMC SNDP]   [ERA5 / TSDPS / IMD]
      (Elevations & Slopes)             (Imperviousness % 2010-2025)     (Pipes, Nalas, Culverts)    (15-Year Rain Hyetographs)
                │                                    │                            │                               │
                └───────────────────┬────────────────┴────────────────────────────┴───────────────────────────────┘
                                    ▼
========================================================================================================================
                                    STAGE 2: HYDRODYNAMIC SIMULATION ENGINE (EPA-SWMM / PySWMM)
========================================================================================================================
                                   ┌────────────────────────────────────────────────────────┐
                                   │ Automated SWMM `.INP` Digital Twin Network Generator   │
                                   │  • Subcatchment Delineation & % Impervious Infiltration│
                                   │  • Pipe Slopes, Diameters, Invert Elevations           │
                                   └───────────────────────────┬────────────────────────────┘
                                                               │
                                                               ▼
                                   ┌────────────────────────────────────────────────────────┐
                                   │ Batch Dynamic Wave Solver (15-Year Continuous + Storms)│
                                   │  • Continuous Monsoons (2010–2025) + IDF Design Storms │
                                   │  • Time-Step: Δt = 1 min (Water Depth H, Flow Q, Flood)│
                                   └───────────────────────────┬────────────────────────────┘
                                                               │
                                                               ▼
========================================================================================================================
                                    STAGE 3: DYNAMIC GRAPH EXTRACTION & DATA PREPROCESSING
========================================================================================================================
  ┌─────────────────────────────────────────────────┐                 ┌────────────────────────────────────────────────┐
  │ Static & Dynamic Node Features X_v(t) ∈ R^{N×Fv}│                 │ Static & Dynamic Edge Features X_e(t) ∈ R^{M×Fe}│
  │ • Rainfall Intensity R(t)                       │                 │ • Pipe Diameter D, Length L, Slope S0          │
  │ • Current Water Depth H(t)                      │                 │ • Manning's Roughness n                        │
  │ • Ground & Invert Elevation (Z_g, Z_inv)        │                 │ • Conduit Flow Rate Q(t)                       │
  │ • Subcatchment Area & Imperviousness %          │                 │ • Filling Ratio (y/D)                          │
  └────────────────────────┬────────────────────────┘                 └───────────────────────┬────────────────────────┘
                           │                                                                  │
                           └────────────────────────────────┬─────────────────────────────────┘
                                                            ▼
                                   ┌────────────────────────────────────────────────────────┐
                                   │ Dynamic Hydraulic Directed Adjacency Matrix A(t)       │
                                   │  • Topological pipe adjacency + Real-time head slope   │
                                   └───────────────────────────┬────────────────────────────┘
                                                               │
                                                               ▼
                                   ┌────────────────────────────────────────────────────────┐
                                   │ Sliding Window Generator & Normalization               │
                                   │  • Input:  Past Tin = 12 steps (60 mins)               │
                                   │  • Target: Future Tout = [t+5m, 15m, 30m, 60m, 120m]   │
                                   │  • Stratified Split: Train (70%), Val (15%), Test (15%)│
                                   └───────────────────────────┬────────────────────────────┘
                                                               │
                                                               ▼
========================================================================================================================
                                    STAGE 4: DEEP LEARNING MODEL ARCHITECTURE (Phy-DSTGNN)
========================================================================================================================
                                   ┌────────────────────────────────────────────────────────┐
                                   │ Input Tensor: X ∈ R^{B × Tin × N × Fv}                  │
                                   └───────────────────────────┬────────────────────────────┘
                                                               │
                                                               ▼
                                   ┌────────────────────────────────────────────────────────┐
                                   │ SPATIO-TEMPORAL ENCODER BLOCKS (Stacked L = 3 layers)  │
                                   │                                                        │
                                   │ 1. Dynamic Directed Spatial GATv2 Layer                │
                                   │    • Message passing along conduits with edge features │
                                   │    • Captures backwater & dynamic surcharge pressure   │
                                   │                                                        │
                                   │ 2. Temporal Dilated Causal Gated Convolutions (TCN)    │
                                   │    • Dilations d ∈ {1, 2, 4} for multi-scale rainfall  │
                                   │                                                        │
                                   │ 3. Temporal Multi-Head Self-Attention                  │
                                   │    • Captures long-range precipitation accumulation    │
                                   └───────────────────────────┬────────────────────────────┘
                                                               │
                                                               ▼
                                   ┌────────────────────────────────────────────────────────┐
                                   │ DUAL-TASK PREDICTION HEADS                             │
                                   │  ├── Regression Head: Future Depth H(t+k) & Flow Q(t+k)│
                                   │  └── Classification Head: Binary Overflow Risk P_flood │
                                   └───────────────────────────┬────────────────────────────┘
                                                               │
                                                               ▼
========================================================================================================================
                                    STAGE 5: PHYSICS-INFORMED MULTI-OBJECTIVE OPTIMIZATION
========================================================================================================================
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Total Loss L_total = L_Huber(Depth) + λ1 * L_Focal(Overflow) + λ2 * L_MassConservation + λ3 * L_ExtremePeak      │
  │                                                                                                                  │
  │ • L_Huber: Smooth regression loss for continuous water levels                                                    │
  │ • L_Focal: Handles extreme class imbalance (floods occur in <3% of steps)                                        │
  │ • L_MassConservation: Penalizes violations of 1D Saint-Venant mass continuity (dVolume/dt - Inflow + Outflow)    │
  │ • L_ExtremePeak: Asymmetric penalty that exponentially punishes under-predicting critical flood peaks            │
  └──────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                             │
                                                             ▼
========================================================================================================================
                                    STAGE 6: MULTI-HORIZON OUTPUTS & REAL-WORLD VALIDATION
========================================================================================================================
  ┌───────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────────────┐
  │ 1. Multi-Horizon Hydrographs      │ 2. Early Warning Alarm System     │ 3. Explainable AI (XAI) Risk Heatmap     │
  │ • Water depth forecast at t+5m,   │ • Binary overflow alert per       │ • Spatial attention identifies           │
  │   t+15m, t+30m, t+60m, t+120m     │   manhole (P_flood > threshold)   │   bottleneck pipes & surcharge origin    │
  │ • Sub-second latency (<40ms)      │ • Critical Lead Time: >30 mins    │ • Overlay with historical flood records  │
  └───────────────────────────────────┴───────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 🗓️ 6-Phase Implementation Roadmap

### Phase 1: Hyderabad 15-Year Data Acquisition & Ingestion (Weeks 1–3)
* **Precipitation (2010–2025):** Download hourly rainfall from ECMWF ERA5-Land, IMD Hyderabad (Begumpet, Shamshabad), and TSDPS Telangana AWS networks.
* **Topography & Elevation:** Download Copernicus 30m DEM / CartoDEM for Hyderabad. Extract junction surface elevations ($Z_{\text{ground}}$) and slopes.
* **15-Year Urbanization Dynamics:** Ingest ESRI 10m / ESA WorldCover land use maps to quantify urbanization expansion across Cyberabad, Gachibowli, and Kukatpally from 2010 to 2025.
* **Drainage Network:** Extract primary/secondary nalas and pipe networks from OpenStreetMap (OSM) and GHMC Strategic Nala Development Programme (SNDP) master plans.

### Phase 2: Hydrologic Parameterization & SWMM Digital Twin Setup (Weeks 4–5)
* Delineate contributing subcatchments for all drainage junctions.
* Assign infiltration parameters (Horton/Green-Ampt) based on Hyderabad soil profiles.
* Construct calibrated `.inp` SWMM model representing Hyderabad's drainage basin (e.g., Kukatpally Nala / Hussain Sagar Basin).

### Phase 3: 15-Year Simulation & Ground Truth Generation (Weeks 6–7)
* Run continuous 15-year monsoon simulations (June–October, 2010–2025) via `PySWMM`.
* Run synthetic stress-testing storms (Hyderabad IDF curves for 2-yr, 5-yr, 10-yr, 25-yr, 50-yr, 100-yr return periods).
* Generate time-synchronized ground-truth datasets for water depth $H_i(t)$, flow rate $Q_{ij}(t)$, and overflow volume $V_{\text{flood}, i}(t)$ at $\Delta t = 1\text{ min}$.

### Phase 4: Dynamic Graph Construction & Imbalance Handling (Weeks 8–9)
* Construct directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A}(t))$ with dynamic edge attention.
* Structure sliding windows: $T_{\text{in}} = 12$ steps (60 min) $\rightarrow$ $T_{\text{out}} = [t+5\text{m}, 15\text{m}, 30\text{m}, 60\text{m}, 120\text{m}]$.
* Implement Focal Loss ($\alpha=0.75, \gamma=2.0$) and Asymmetric Extreme Peak Loss to solve the $<3\%$ flood class imbalance.

### Phase 5: Deep Learning Model Training & Optimization (Weeks 10–12)
* Implement and train the proposed **Phy-DSTGNN** architecture.
* Optimize the physics-informed mass conservation loss penalty ($\mathcal{L}_{\text{mass}}$).
* Benchmark against state-of-the-art baselines: Random Forest, XGBoost, Vanilla LSTM, BiLSTM, CNN-LSTM, Transformer (PatchTST), STGCN, and DCRNN.

### Phase 6: Validation, Explainability & Manuscript Preparation (Weeks 13–15)
* Benchmark specifically on the **October 13–14, 2020 Hyderabad Super-Flood (320 mm/24h)**.
* Generate Explainable AI (XAI) attention heatmaps revealing network bottlenecks.
* Draft full technical research manuscript with high-resolution vector figures and metric tables.
