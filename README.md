<div align="left">
  <img src="assets/AppLogo.svg" alt="Jennifer Logo" width="300"/>
  <br><br>

  [![CI](https://github.com/sarthak-shubham/jennifer/actions/workflows/ci.yml/badge.svg)](https://github.com/sarthak-shubham/jennifer/actions/workflows/ci.yml)
  <img src="https://img.shields.io/badge/Event-42nd_Cloudflight_CCC-00c853?style=flat-square" alt="Event"/>
  <img src="https://img.shields.io/badge/Track-Data%20%2F%20AI%20Edition-00b0ff?style=flat-square" alt="Track"/>
  <img src="https://img.shields.io/badge/Model-Jennifer_CNN_v3-7c4dff?style=flat-square" alt="Model"/>
  <img src="https://img.shields.io/badge/Footprint-23.6_KB_ONNX-ff6d00?style=flat-square" alt="Size"/>
  <img src="https://img.shields.io/badge/Opset-18_(IR_9)-d500f9?style=flat-square" alt="Opset"/>
  <img src="https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
</div>

---

## Overview

Jennifer is a tactical game engine and neural evaluator developed for the **42nd Cloudflight Coding Contest (Data / AI Edition)**, competing within the **Colony: One** environment.

The codebase implements two complementary systems:
1. **Deterministic Solvers (Levels 1 & 2):** Exact state-space engines providing mathematical guarantees for collision avoidance and forced-win extraction.
2. **Policy-Value Evaluator (Levels 3 & 4):** A hand-engineered 23.6 KB Convolutional Neural Network exported to ONNX format, designed for sub-millisecond evaluation during live Monte Carlo Tree Search (MCTS) rollouts.

---

## Certification

The entry was developed and evaluated during the live competition:

* **Official Participation Document:** [participation_certificate.pdf](assets/participation_certificate.pdf)
* **Event:** 42nd Classic Cloudflight Coding Contest (Data / AI Arena)

---

## Problem Coverage

The contest was organized into four progressive levels, shifting from local geometric constraints to global adversarial search:

* **Level 1 (Safe Navigation):** Given positions for a friendly Kernel, enemy Kernel, and optional enemy Ping, enumerate all two-step move sequences that remain uncapturable on the opponent's subsequent turn.
* **Level 2 (Combinatorial Search):** Multi-piece board configurations requiring the calculation of the shortest forced-win line, accounting for the Bodyguard escort mechanic.
* **Level 3 (Contract & Local Validation):** Introduction of the full game simulation. Submissions required exporting an ONNX model that satisfied strict schema, latency, and memory budgets under local MCTS evaluation.
* **Level 4 (Tournament Arena):** Uploaded models were deployed to a live multiplayer server, competing in concurrent matches against hundreds of participant-submitted engines.

---

## Implementation

The implementation is divided into deterministic pathfinding and learned board valuation.

### Solvers

* **Pathfinder (`solvers/level1_safe_pathfinder.py`):** Formulates move generation as a breadth-first search across the two-step action space, verifying that candidate final positions fall outside the Chebyshev radius of the opponent Kernel and the raycast line-of-fire of the opponent Ping.
* **Minimax Engine (`solvers/level2_minimax_solver.py`):** Explores the game tree using minimax search with transposition memoization and iterative deepening. Correctly evaluates the Bodyguard mechanic: an attack on an escorted Kernel sacrifices the escort rather than capturing the Kernel.

### Model Architecture & Evolution

To avoid the memory and runtime overhead of standard training frameworks, the model graph was synthesized directly using ONNX operator primitives:

* **v1 Linear Baseline (`model_builders/experiments/v1_linear_baseline.py`):** Evaluates board states via static material balance matrices and center-ring positional bonuses (12.8 KB).
* **v2 Feature Broadcaster (`model_builders/experiments/v2_feature_broadcaster.py`):** Introduces perimeter evacuation urgency to anticipate the shrinking firewall boundary (12.8 KB).
* **v3 Tactical ConvNet (`model_builders/build_jennifer_cnn.py`):** The production tournament architecture. Couples a single $3 \times 3$ spatial convolution layer (8 channels) with factorized source and destination policy scoring (23.6 KB).

For the complete technical specification—including the 16-plane state encoding, convolution filter weight kernels, and dual-head matrix factorization—see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Setup & Execution

### Prerequisites

* Python 3.11+
* Install package dependencies in editable mode:
  ```bash
  pip install -e .
  ```

### Commands

Compile the production neural network:
```bash
python model_builders/build_jennifer_cnn.py
```

Run the competition acceptance validator:
```bash
python tools/validate_submission.py models/jennifer_tournament.onnx
```

Execute a 10-game simulation against the official random baseline:
```bash
python tools/model_vs_model.py --model-a models/jennifer_tournament.onnx --model-b models/random_baseline.onnx --games 10
```

Execute the deterministic puzzle solvers:
```bash
python solvers/level1_safe_pathfinder.py E4 H4 A4
python solvers/level2_minimax_solver.py C3 F5 F8
```

### Continuous Integration

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) triggered on push and manual dispatch (`workflow_dispatch`). The workflow executes on an Ubuntu runner to build the model, verify submission contract compliance, execute test states against both solvers, and run an end-to-end benchmark against the baseline engine.

---

## Live Tournament Telemetry

During Level 4, the model was uploaded directly to the platform evaluation cluster.

<details>
<summary>View Level 4 Tournament Dashboard</summary>

<br>

[![Cloudflight Tournament Dashboard](assets/dashboard_view.png)](assets/dashboard_view.png)

</details>

<details>
<summary>View Match Recording</summary>

<br>

* Direct media file: [assets/match_recording.mp4](assets/match_recording.mp4) (25.6 MB)

</details>

### Telemetry Summary

| Metric | Platform Value | Details |
| :--- | :---: | :--- |
| **Registered Kernels** | 293 models | Active competing engines |
| **Total Arena Matches** | 23,782 games | Concurrent multi-agent games evaluated |
| **Matches per Minute** | 91.80 | Server evaluation throughput |
| **Deployed Binary Size** | 0.02 MiB (23.6 KB) | Within the 5.0 MB platform ceiling |
| **Levels Cleared** | 1, 2, 3, 4 | Full progression across all contest tiers |

---

## What I Learned

* **Inference Latency in MCTS:** Turn-based game engines with tight deadlines (2.0s per turn) place an extreme premium on evaluation throughput. A 23.6 KB model executing in under 0.2 ms allows the MCTS search to explore significantly more nodes per second than deeper networks, resulting in superior tactical play under time pressure.
* **Direct Graph Synthesis:** Constructing computation graphs directly with `onnx.helper` eliminates unnecessary runtime dependencies and guarantees deterministic execution characteristics across evaluation environments.
* **Constraint-Driven Design:** Progressing from exact search to heuristic evaluation reinforced the trade-offs between mathematical certainty in restricted domains and scalable approximation in large state spaces.

---

## Author

* **Sarthak Shubham** — [GitHub](https://github.com/sarthak-shubham)
