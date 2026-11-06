# Architecture Reference: Jennifer Tactical ConvNet

Specification for the model compiled by `model_builders/build_jennifer_cnn.py`.

---

## State Encoding

Board states are encoded as `(batch, 16, 8, 8)` float32 tensors from the perspective of the team to move. When Magenta moves, the board is rotated 180° so the network always sees its own pieces in planes 0–5 and enemy pieces in planes 6–11.

| Plane | Contents |
|:---:|:---|
| 0 | Friendly Ping |
| 1 | Friendly Daemon |
| 2 | Friendly Exploit |
| 3 | Friendly Trace |
| 4 | Friendly Ice |
| 5 | Friendly Kernel |
| 6 | Enemy Ping |
| 7 | Enemy Daemon |
| 8 | Enemy Exploit |
| 9 | Enemy Trace |
| 10 | Enemy Ice |
| 11 | Enemy Kernel |
| 12 | Firewall-locked squares (1.0 = impassable) |
| 13 | Actions remaining this turn (`actions_left / 2`; either `0.5` or `1.0`) |
| 14 | Firewall urgency (scales toward `1.0` as the next ring collapse approaches) |
| 15 | Side to move (`1.0` = Magenta, `0.0` = Cyan) |

Plane 15 is required because the game is asymmetric: Magenta takes every tiebreak and the firewall only shrinks after Magenta's turn. Without it, two positions with different values could produce identical tensors.

---

## Convolution Layer

A single 3×3 convolution with 8 output channels and ReLU activation, applied with padding 1 to preserve the 8×8 spatial dimensions:

$$\mathbf{F} = \text{ReLU}\left( \text{Conv2D}(\mathbf{X}, \mathbf{W}) \right) \in \mathbb{R}^{B \times 8 \times 8 \times 8}$$

Bias is zero. Stride is 1.

### Filters

| Filter | Role | Planes Used | Weights |
|:---:|:---|:---|:---|
| 0 | Friendly Kernel bodyguard detection | Plane 5 (center), Planes 0–4 (neighbors) | Center: `+1.0`, neighbors: `+0.25` |
| 1 | Enemy Kernel bodyguard detection | Plane 11 (center), Planes 6–10 (neighbors) | Center: `+1.0`, neighbors: `+0.25` |
| 2 | Enemy threat zone | Plane 7 (Daemon), Plane 11 (Kernel) adjacent | Daemon: `+1.0`, Kernel: `+0.5` |
| 3 | Friendly Daemon line-of-fire | Plane 1 (Daemon) adjacent | `+1.0` |
| 4 | Local material balance | Planes 0–5 vs 6–11 at center | Friendly: positive, enemy: negative |
| 5 | Center ring control | Plane 5 (Friendly Kernel) | `+0.2` across full 3×3 |
| 6 | Actions-remaining modulation | Plane 13 at center | `+1.0` (activates on first action, `0.5`; both actions, `1.0`) |
| 7 | Threat to friendly Kernel | Planes 6–11 adjacent | `+0.5` per enemy piece |

---

## Value and Policy Heads

The conv output `F` is concatenated with the raw input `X` along the channel axis, producing a 24-channel tensor `Z` of shape `(B, 24, 8, 8)`.

### Value Head

`Z` is flattened to `(B, 1536)` and projected through a linear layer followed by tanh:

$$V(s) = \tanh\left( \mathbf{Z}_{\text{flat}} \cdot \mathbf{W}_{\text{val}} \right) \in [-1, 1]$$

Weights assign higher values to friendly material and escort formations, and penalize enemy material and exposure of the friendly Kernel.

### Policy Head

The full action space is 64 × 64 = 4096 source-to-destination pairs. A full `1536 × 4096` matrix would exceed the 25 KB size budget, so the policy is factorized:

$$\text{Policy}(u, v) = \text{Score}_{\text{from}}(u) + \text{Score}_{\text{to}}(v)$$

- **`Score_to(v)`** — rewards capturing high-value targets (`+30.0` Kernel, `+15.0` Daemon) and penalizes squares covered by Filter 2 (`-6.0`).
- **`Score_from(u)`** — rewards activating attacking pieces and penalizes pieces on the outer ring before firewall collapse (`+2.5` perimeter penalty).

---

## Model Versions

| Version | File | Size | Change |
|:---|:---|:---:|:---|
| v1 | `model_builders/experiments/v1_linear_baseline.py` | 12.8 KB | Material weight matrix with center ring bonus |
| v2 | `model_builders/experiments/v2_feature_broadcaster.py` | 12.8 KB | Added firewall urgency and perimeter evacuation penalty |
| **v3** | `model_builders/build_jennifer_cnn.py` | **23.6 KB** | 3×3 spatial convolution with factorized policy head |

---

## Platform Compliance

| Property | Value |
|:---|:---|
| Opset | 18 |
| IR Version | 9 |
| Batch axis | Dynamic across all input and output tensors |
| Binary size | 23.61 KB (platform limit: 5.0 MB) |
| Inference latency | < 0.2 ms on single-thread CPU |
