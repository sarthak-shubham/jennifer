# 🛸 Jennifer — Autonomous Tactical Game AI & Adversarial Engine

[![Competition](https://img.shields.io/badge/Competition-42nd%20Cloudflight%20Coding%20Contest%20(Data%2FAI)-blue.svg)](https://codingcontest.org)
[![Status](https://img.shields.io/badge/Stage-Level%204%20Tournament%20Finalist-gold.svg)](#tournament-results)
[![Model](https://img.shields.io/badge/ONNX-Opset%2018%20%7C%2023.6%20KB-success.svg)](#jennifer-neural-architecture)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Jennifer** is an autonomous tactical game AI, adversarial search engine, and custom neural policy-value model developed for the **42nd Cloudflight Coding Contest (Data/AI Edition)** on the cyber-themed strategy environment ***Colony: One***.
>
> Successfully qualified through all algorithmic puzzle rounds (Levels 1–3) to reach the **Level 4 Live Cloud Tournament**, competing in real-time pairing rounds among **330+ global entries**.

---

## 🎮 The Game: *Colony: One*

*Colony: One* is a turn-based, asymmetric multi-agent strategy board game on an $8 \times 8$ grid:
* **2-Action Turns:** Each team plays **exactly two consecutive actions** per turn before the turn passes to the opponent. Action 1 often serves as tactical positioning/ambush setup, while Action 2 delivers the strike or defensive retreat.
* **The Centerpiece (Kernel):** Losing your Kernel is instant defeat; capturing the enemy Kernel is instant victory.
* **The Bodyguard Rule:** A Kernel with an adjacent friendly piece gains an **invincibility shield**. When attacked, the protecting bodyguard is sacrificed instead of the Kernel.
* **Firewall:** The board shrinks over rounds ($8 \times 8 \to 6 \times 6 \to 4 \times 4 \to 2 \times 2$), permanently destroying pieces on newly locked rings.
* **Endgame Resolution:** When only Kernels remain, the player occupying the more central ring wins; ties favor Magenta.

### Pieces
| Piece | Movement | Attack / Capture Mechanics |
| :--- | :--- | :--- |
| **Kernel** | 1 step in any direction | Move-captures onto target |
| **Ping** | Orthogonal up to 2 tiles (empty only) | Diagonally captures enemy within 2 tiles |
| **Daemon** | Orthogonal slide up to 3 tiles | Shoots adjacent enemies without moving; move-captures at range 2–3 |
| **Exploit** | $2+1$ L-shaped jump (Knight) | Leaps over barriers; move-captures onto target |
| **Trace** | 1 orthogonal step | Snipes diagonal enemies at range 2–3 without moving |
| **Ice** | Up to 2 tiles orthogonal or diagonal | Move-captures onto target |

---

## 🧠 System Architecture & Progression

```
Level 1: In a Pinch           Level 2: New Friends          Level 3 & 4: Live Tournament
┌────────────────────┐        ┌────────────────────┐        ┌────────────────────────────┐
│ Safe-Tile Analysis │   ──>  │ Adversarial Minimax│   ──>  │ Jennifer CNN Model (ONNX)   │
│ & 2-Action BFS     │        │ Game-Tree Engine   │        │ + Cloud MCTS (100 rollouts)│
└────────────────────┘        └────────────────────┘        └────────────────────────────┘
```

### 1. Level 1 — Survival Pathfinding
* Implemented graph-search state validation evaluating two-action movement branches against opponent threat cones to identify guaranteed safe landing tiles.

### 2. Level 2 — Minimax Adversarial Engine
* Built an adversarial minimax engine with depth search that models the full 2-action turn mechanics, bodyguard absorption logic, and corner-trapping algorithms to deliver forced checkmates in minimum turns.

### 3. Levels 3 & 4 — The Jennifer Neural Network
In the final tournament stage, the server referee runs **Monte Carlo Tree Search (MCTS)** with 100 simulations per action, querying an ONNX neural model for:
1. **Policy Logits ($4096$):** Prior probabilities $P(s, a)$ over all possible $(from\_square \times 64 + to\_square)$ actions.
2. **Position Value ($[-1, 1]$):** Leaf evaluation $V(s)$ representing side-to-move win probability.

#### Key Neural Innovations:
* **$3 \times 3$ Spatial Convolution Layer (`Conv`):**
  * **Bodyguard Shield Detection:** Convolves adjacent friendly non-Kernel pieces around the Kernel to dynamically output a **+1.5 survival bonus**.
  * **Vulnerable Kernel Flag:** Highlights an enemy Kernel with zero guards as an immediate assassination target (**+30.0 policy logit**).
  * **Daemon Threat Zone:** Maps range-1 adjacent kill zones.
* **Blunder Prevention & Threat Mapping:**
  * Computes an enemy danger crossfire map, heavily penalizing moves that step into free enemy fire (**-6.0**), eliminating "hanging pieces."
* **Asymmetric Action Modulation (Plane 13):**
  * Modulates tactical behavior based on `actions_left / 2`: prioritizes ambush setup on Action 1 and strikes/retreats on Action 2.
* **Production Constraints:**
  * **Size:** Only **23.62 KB** (tournament limit: 5 MiB).
  * **Latency:** **< 0.2 ms** CPU inference (tournament limit: 3 seconds).
  * **Compatibility:** ONNX opset 18, `ir_version = 9`, dynamic batch axis.

---

## 🏆 Tournament Results

* **Stage Reached:** **Level 4 Live Tournament** (Finalist)
* **Matches Played:** 100+ live pairing rounds against global submissions.
* **Global Contest Scope:** 330+ participating teams and solo developers worldwide.
* **Official Verification:** Certified by Cloudflight GmbH (42nd Data/AI CCC).

---

## 🛠️ Repository Structure

```text
jennifer/
├── README.md                           # Project documentation & tournament breakdown
├── model.onnx                          # Production-deployed CNN model (23.6 KB)
├── solver.py                           # Level 1 safe-square pathfinder
├── game_solver.py                      # Level 2 adversarial minimax solver
├── level-3/
│   ├── build_conv_tactical_model.py    # CNN generator script (Opset 18, IR 9)
│   ├── colony_one/                     # Tournament environment & MCTS referee
│   └── tools/validate_submission.py    # Local submission validator
├── level-1/level-1.pdf                 # Problem specification: Level 1
├── level-2/level-2.pdf                 # Problem specification: Level 2
├── level-3/level-3.pdf                 # Problem specification: Level 3
├── level-4/level-4.pdf                 # Problem specification: Level 4
└── ccc-participation-certificate...pdf # Official CCC participation certificate
```

---

## 🚀 Quickstart

### Prerequisites
* Python 3.11+
* `pip install onnx numpy`

### Run Model Builder
To compile and export the Jennifer ONNX neural network:
```bash
python level-3/build_conv_tactical_model.py
```

### Run Minimax Puzzle Solver
To calculate exact forced-win lines from any board state:
```bash
python game_solver.py <CYAN_KERNEL> <CYAN_PING> <MAGENTA_KERNEL>
# Example: python game_solver.py C3 F5 F8
```

---

## 👤 Author

* **Sarthak Shubham**
  * Competition: 42nd Cloudflight Coding Contest (Data/AI)
  * Date: October 2026
