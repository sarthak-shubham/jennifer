# Technical Specification: Jennifer Tactical ConvNet

This document specifies the internal state representation, convolutional filter weights, and dual-head factorization for the model compiled by `model_builders/build_jennifer_cnn.py`.

---

## State Representation

Game states are mapped into a single-precision floating point tensor of shape `(batch, 16, 8, 8)` representing 16 spatial planes:

| Plane | Identifier | Definition |
| :---: | :--- | :--- |
| **0 – 4** | Friendly Units | Spatial bitmasks for friendly Ping (0), Daemon (1), Exploit (2), Trace (3), and Ice (4) |
| **5** | Friendly Kernel | The primary unit. Capture of this piece results in immediate defeat |
| **6 – 10** | Enemy Units | Spatial bitmasks for enemy Ping (6), Daemon (7), Exploit (8), Trace (9), and Ice (10) |
| **11** | Enemy Kernel | Target piece for forced checkmate |
| **12** | Actions Remaining | Scalar plane indicating action count remaining in current turn |
| **13** | Action Phase | `0.0` on Action 1; `1.0` on Action 2 |
| **14** | Firewall Urgency | Scalar plane scaling with turns remaining before the next perimeter collapse |
| **15** | Firewall Boundary | Spatial mask of scorched and impassable tiles |

---

## Convolutional Filter Specifications

The architecture applies a single $3 \times 3$ convolution layer with 8 output channels, followed by a ReLU activation function:

$$\mathbf{F} = \text{ReLU}\left( \text{Conv2D}(\mathbf{X}, \mathbf{W}_{\text{conv}}) + \mathbf{b} \right) \in \mathbb{R}^{\text{batch} \times 8 \times 8 \times 8}$$

Where:
* Padding: 1 on all borders (preserves spatial $8 \times 8$ dimensionality)
* Stride: 1
* Bias: $\mathbf{b} = \mathbf{0}$

### Filter Descriptions

* **Filter 0 (Friendly Escort Detection):** Centered at $(1, 1)$ on Plane 5 with uniform weights of $+0.25$ across the 8 adjacent neighbors on friendly unit planes (0..4). Activates when the friendly Kernel satisfies the Bodyguard condition.
* **Filter 1 (Enemy Escort Detection):** Centered at $(1, 1)$ on Plane 11 with uniform weights of $+0.25$ across the 8 adjacent neighbors on enemy unit planes (6..10). Distinguishes escorted enemy Kernels from isolated targets.
* **Filter 2 (Enemy Attack Zone):** Evaluates Daemon (Plane 7, weight 1.0) and Kernel (Plane 11, weight 0.5) attack corridors across adjacent tiles. Used by the policy head to penalize moves entering immediate lines of fire.
* **Filter 3 (Friendly Line-of-Fire):** Maps squares adjacent to friendly Daemons (Plane 1, weight 1.0) to prioritize captures and tactical shots.
* **Filter 4 (Local Material Balance):** Differential $3 \times 3$ window comparing friendly piece weights against enemy piece weights per square.
* **Filter 5 (Center Ring Control):** Center-weighted mask applied to Plane 5 to favor interior over boundary placement.
* **Filter 6 (Action Phase Modulation):** Scales with Plane 13 to differentiate Action 1 offensive setup from Action 2 defensive consolidation.
* **Filter 7 (Kernel Danger Map):** Identifies enemy units directly adjacent to the friendly Kernel.

---

## Policy and Value Heads

Feature tensor $\mathbf{F}$ is concatenated with the raw input tensor $\mathbf{X}$ along the channel dimension, yielding a 24-channel tensor $\mathbf{Z} \in \mathbb{R}^{\text{batch} \times 24 \times 8 \times 8}$.

### Value Head

$\mathbf{Z}$ is flattened to $(B, 1536)$ and projected through a single linear transformation with hyperbolic tangent activation:

$$V(s) = \tanh\left( \mathbf{Z}_{\text{flat}} \mathbf{W}_{\text{val}} \right) \in [-1, 1]$$

The weight matrix $\mathbf{W}_{\text{val}}$ assigns positive weights to friendly material dominance and escort formations (+1.5), and negative weights to enemy material and Kernel exposure (-3.0).

### Policy Head

The action space consists of $64 \times 64 = 4096$ source-to-destination pairs. To maintain an ONNX binary size under 25 KB without a full $1536 \times 4096$ parameter matrix, the policy is factorized into additive source and destination scores:

$$\text{Policy}(u, v) = \text{Score}_{\text{from}}(u) + \text{Score}_{\text{to}}(v)$$

* $\text{Score}_{\text{to}}(v)$: Computed across the 24 channels for destination square $v$. Rewards landing on high-value enemy targets (+30.0 for Kernel, +15.0 for Daemon) and penalizes landing on squares active in Filter 2 (-6.0 blunder prevention).
* $\text{Score}_{\text{from}}(u)$: Evaluates source square $u$. Incentivizes activating attack pieces and penalizes pieces lingering on the perimeter before firewall contraction (+2.5 perimeter penalty).

---

## Model Evolution History

| Version | File | Size | Key Technique |
| :--- | :--- | :---: | :--- |
| **v1 (Linear)** | `model_builders/experiments/v1_linear_baseline.py` | 12.8 KB | Material weighting matrix with center ring bonus |
| **v2 (Broadcaster)** | `model_builders/experiments/v2_feature_broadcaster.py` | 12.8 KB | Added firewall urgency modulation and piece targeting |
| **v3 (ConvNet)** | `model_builders/build_jennifer_cnn.py` | **23.6 KB** | $3 \times 3$ spatial convolution with factorized policy head |

---

## Platform Compliance

* **Opset Version:** 18
* **IR Version:** 9 (satisfies competition ceiling $\le 13$)
* **Batching:** Dynamic batch axis across input and all output tensors
* **Inference Latency:** $< 0.2 \text{ ms}$ on single-thread CPU
* **Binary Size:** 23.61 KB
