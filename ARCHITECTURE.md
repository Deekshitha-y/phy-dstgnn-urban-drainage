# Model Architecture: Physics-Informed Dynamic Spatio-Temporal Graph Neural Network (Phy-DSTGNN)

---

## 1. Mathematical Formulation

### 1.1 Drainage Network as a Directed Dynamic Graph
Let the urban drainage network be defined as a directed spatio-temporal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A}(t))$:
* **Nodes $\mathcal{V}$ ($N$ junctions/manholes):**
  $$\mathbf{X}_v^{(t)} \in \mathbb{R}^{N \times F_v}, \quad \mathbf{X}_{v, i}^{(t)} = \Big[ R_i(t), \; H_i(t), \; Z_{\text{ground}, i}, \; Z_{\text{invert}, i}, \; A_{\text{sub}, i}, \; C_{\text{imperv}, i}(t) \Big]$$
* **Edges $\mathcal{E}$ ($M$ conduits/pipes):**
  $$\mathbf{X}_e^{(t)} \in \mathbb{R}^{M \times F_e}, \quad \mathbf{X}_{e, ij}^{(t)} = \Big[ L_{ij}, \; D_{ij}, \; S_{0, ij}, \; n_{ij}, \; Q_{ij}(t) \Big]$$
* **Dynamic Hydraulic Edge Attention Matrix $\mathbf{A}(t) \in \mathbb{R}^{N \times N}$:**
  $$\mathbf{A}_{ij}(t) = \text{Softmax}\left( \mathbf{W}_a \left[ \mathbf{X}_i(t) \parallel \mathbf{X}_j(t) \parallel \mathbf{X}_{e, ij}(t) \right] + \beta \frac{H_i(t) - H_j(t)}{L_{ij}} \right)$$

---

## 2. Layer-by-Layer Network Structure

### 2.1 Spatial Module: Directed GATv2 with Dynamic Edge Attributes
To capture directional pipe flow, dynamic surcharge, and reverse backwater pressure:
$$\mathbf{z}_{ij}^{(t)} = \text{LeakyReLU}\left( \mathbf{a}^T \left[ \mathbf{W}_n \mathbf{h}_i^{(t)} \parallel \mathbf{W}_n \mathbf{h}_j^{(t)} \parallel \mathbf{W}_e \mathbf{X}_{e, ij}^{(t)} \right] \right)$$
$$\alpha_{ij}^{(t)} = \frac{\exp(\mathbf{z}_{ij}^{(t)})}{\sum_{k \in \mathcal{N}_{\text{in}}(i) \cup \mathcal{N}_{\text{out}}(i)} \exp(\mathbf{z}_{ik}^{(t)})}$$
$$\mathbf{h}_{i, \text{spatial}}^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(t)} \mathbf{W}_v \mathbf{h}_j^{(l)} \right)$$

### 2.2 Temporal Module: Dilated Causal Gated Convolutions (TCN)
To capture multi-scale rainfall accumulation without recurrence bottlenecks:
$$\mathbf{\Gamma}_{\text{temporal}} = \tanh\left( \mathbf{\Theta}_f \ast_d \mathbf{H}_{\text{spatial}} \right) \odot \sigma\left( \mathbf{\Theta}_g \ast_d \mathbf{H}_{\text{spatial}} \right)$$
* Dilations: $d \in \{1, 2, 4\}$ for receptive fields covering $5\text{ min}$ to $120\text{ min}$.

### 2.3 Temporal Multi-Head Self-Attention (MHSA)
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

---

## 3. Physics-Informed Multi-Objective Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Huber}}(\hat{\mathbf{H}}, \mathbf{H}) + \lambda_1 \mathcal{L}_{\text{focal}}(\hat{\mathbf{P}}, \mathbf{y}) + \lambda_2 \mathcal{L}_{\text{mass}} + \lambda_3 \mathcal{L}_{\text{peak}}$$

### 3.1 1D Saint-Venant Mass Conservation Loss ($\mathcal{L}_{\text{mass}}$)
Enforces continuity at every junction $i$:
$$\mathcal{L}_{\text{mass}} = \frac{1}{N} \sum_{i=1}^N \left| A_{\text{manhole}, i} \frac{\hat{H}_i(t+\Delta t) - \hat{H}_i(t)}{\Delta t} - \left( \sum_{j \in \mathcal{N}_{\text{in}}(i)} \hat{Q}_{ji}(t) - \sum_{k \in \mathcal{N}_{\text{out}}(i)} \hat{Q}_{ik}(t) + Q_{\text{runoff}, i}(t) \right) \right|$$

### 3.2 Asymmetric Extreme Peak Loss ($\mathcal{L}_{\text{peak}}$)
Penalizes under-prediction of extreme flood peaks:
$$\mathcal{L}_{\text{peak}} = \sum_{i=1}^N \mathbb{I}(H_i(t) > H_{\text{threshold}, i}) \cdot \exp\left( \frac{\hat{H}_i(t) - H_i(t)}{H_{\text{ground}, i} - Z_{\text{invert}, i}} \right)^2$$

### 3.3 Focal Loss for Rare Overflow Classification ($\mathcal{L}_{\text{focal}}$)
$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \alpha = 0.75, \; \gamma = 2.0$$

---

## 4. Tensor Dimensions & Data Pipeline

```
[Input Tensor: B × 12 × N × 6]
         │
         ▼
[Spatial GATv2 Embedding: B × 12 × N × 64]
         │
         ▼
[Dilated TCN + Temporal Attention: B × 12 × N × 128]
         │
         ├────────────────────────────────────────┬────────────────────────────────────────┐
         ▼                                        ▼                                        ▼
[Regression Head (Depth): B × 5 × N]     [Regression Head (Flow): B × 5 × M]     [Classification Head: B × 5 × N]
(Continuous Depth H at +5 to +120 min)   (Conduit Flow Q at +5 to +120 min)      (Binary Surcharge Alert P ∈ [0, 1])
```
