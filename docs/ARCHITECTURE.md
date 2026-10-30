# 🧠 Neural Architecture & Technical Specification

This document details the mathematical formulation, tensor representations, and spatial convolution filter design of the **Jennifer Tactical ConvNet** (`jennifer_tournament.onnx`), developed for the *Colony: One* multi-agent tournament.

---

## 📐 Observation Tensor Representation

The game state is encoded into a canonical float32 tensor of shape `(batch, 16, 8, 8)` from the perspective of the current side to move:

| Plane Index | Identity | Semantic Meaning |
| :---: | :--- | :--- |
| **0 – 4** | Friendly Non-Kernel | Ping (0), Daemon (1), Exploit (2), Trace (3), Ice (4) |
| **5** | Friendly Kernel | The primary VIP piece. If captured, team loses immediately |
| **6 – 10** | Enemy Non-Kernel | Enemy Ping (6), Daemon (7), Exploit (8), Trace (9), Ice (10) |
| **11** | Enemy Kernel | Target piece for forced checkmate |
| **12** | Friendly Turns Left | Binary mask indicating remaining actions in current turn |
| **13** | Action Phase | `0.0` for Action 1; `1.0` for Action 2 |
| **14** | Firewall Urgency | Scalar plane scaling with turns remaining until the next perimeter collapse |
| **15** | Firewall Burn Mask | Spatial boolean mask of destroyed / scorched tiles |

---

## 🔍 The 8 Spatial Convolution Filters

Jennifer applies a single $3 \times 3$ convolution layer ($C_{\text{in}} = 16$, $C_{\text{out}} = 8$, $\text{padding} = 1$, $\text{stride} = 1$) followed by a `ReLU` activation:

$$\mathbf{F} = \text{ReLU}\left( \text{Conv2D}(\mathbf{X}, \mathbf{W}_{\text{conv}}) + \mathbf{b} \right) \in \mathbb{R}^{\text{batch} \times 8 \times 8 \times 8}$$

### Filter Specification

```
Filter 0: Friendly Bodyguard Detection
Kernel centered at (1, 1) on Plane 5.
Surrounding 8 squares weighted (+0.25) across friendly planes (0..4).
Detects whether the friendly Kernel is actively shielded by friendly escorts.

Filter 1: Enemy Bodyguard Detection
Enemy Kernel centered at (1, 1) on Plane 11.
Surrounding 8 squares weighted (+0.25) across enemy planes (6..10).
Flags whether the enemy Kernel is escorted or vulnerable to a direct assassination.

Filter 2: Enemy Threat Zone (Danger Map)
Weights Daemon (Plane 7, weight 1.0) and Kernel (Plane 11, weight 0.5) across all 8 adjacent squares.
Used by the policy head to penalize moves landing in immediate capture lines (blunder prevention).

Filter 3: Friendly Daemon Kill Zone
Surrounding squares centered on friendly Daemon (Plane 1) weighted at +1.0.
Rewards herding enemy pieces into adjacent point-blank shooting range.

Filter 4: Local Material Dominance
Differential 3x3 kernel comparing friendly piece values vs. enemy piece values centered on each tile:
- Friendly: +0.5 (Ping), +1.5 (Daemon), +1.0 (Exploit), +1.0 (Trace), +1.5 (Ice), +5.0 (Kernel)
- Enemy: Negative reciprocal weights.

Filter 5: Center Ring Control
Weights the center 3x3 region on Plane 5 (+0.2) to maintain central dominance.

Filter 6: Action Phase Modulation
Coupled with Plane 13 to modulate aggressive forward postures on Action 1 vs. defensive retreats on Action 2.

Filter 7: Kernel Under Attack Map
Enemy pieces (Planes 6..11) immediately adjacent to the friendly Kernel (Plane 5).
Triggers highest priority evacuation and defensive counter-strikes.
```

---

## 🎯 Dual-Head Architecture

The raw observation $\mathbf{X} \in \mathbb{R}^{B \times 16 \times 8 \times 8}$ is concatenated with the extracted spatial features $\mathbf{F} \in \mathbb{R}^{B \times 8 \times 8 \times 8}$ along the channel axis:

$$\mathbf{Z} = \text{Concat}(\mathbf{X}, \mathbf{F}, \text{axis}=1) \in \mathbb{R}^{B \times 24 \times 8 \times 8}$$

### 1. Value Head $V(s) \in [-1, 1]$
$\mathbf{Z}$ is flattened to $(B, 1536)$ and projected via a linear matrix multiplication into a scalar, followed by a hyperbolic tangent activation:

$$V(s) = \tanh\left( \mathbf{Z}_{\text{flat}} \mathbf{W}_{\text{val}} \right) \in [-1, 1]$$

* Material weighting evaluates piece advantages.
* Spatial features assess bodyguard protection (+1.5) and penalize enemy threats to our Kernel (-3.0).

### 2. Policy Head $P(s) \in \mathbb{R}^{4096}$
The action space consists of $64 \times 64 = 4096$ possible origin-destination square pairs. To preserve an ultra-compact footprint without requiring a massive dense layer, the policy is factorized into source and destination scores:

$$\text{Policy}(u, v) = \text{Score}_{\text{from}}(u) + \text{Score}_{\text{to}}(v)$$

Where:
* $\text{Score}_{\text{to}}(v)$: Evaluates destination squares. Heavily rewards capturing enemy high-value targets (+30.0 for Kernel, +15.0 for Daemon) and penalizes landing on squares in Filter 2's enemy threat zone (-6.0 blunder prevention).
* $\text{Score}_{\text{from}}(u)$: Evaluates source squares. Incentivizes activating agile attack pieces and penalizes pieces lingering on outer perimeter rings before firewall sweeps (+2.5 urgency penalty).

---

## 📊 Evolutionary Iterations

| Iteration | File | Params / Size | Key Mechanism |
| :--- | :--- | :---: | :--- |
| **v1 (Linear)** | `model_builders/experiments/v1_linear_baseline.py` | 12.8 KB | Material matrix + static Chebyshev center bonus |
| **v2 (Broadcaster)** | `model_builders/experiments/v2_feature_broadcaster.py` | 12.8 KB | Added firewall urgency plane modulation & piece targeting |
| **v3 (Jennifer CNN)** | `model_builders/build_jennifer_cnn.py` | **23.6 KB** | 8-Channel $3 \times 3$ Conv Layer + Factorized Policy Head |

---

## ⚙️ Submission Contract Compliance

The model is compiled natively into ONNX with:
* **Opset Version:** 18
* **IR Version:** 9 (strictly $\le 13$ for compatibility with tournament grading servers)
* **Dynamic Batching:** Dimension 0 on `observation`, `policy`, and `value` are dynamic (`batch`)
* **Inference Latency:** $< 0.2 \text{ ms}$ on single-thread CPU
* **Total File Size:** $23.61 \text{ KB}$ (well under the 5.0 MB platform limit)
