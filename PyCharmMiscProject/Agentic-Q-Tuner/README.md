# Agentic-Q-Tuner

**A demonstration of proficiency in high-dimensional omics data analysis and agentic AI automation.**

This project implements a Quantum-Inspired Evolutionary Algorithm (QEA) for biomarker discovery in a synthetic 50-sample × 5,000-gene dataset. An autonomous `AgentManager` drives the evolution in real time, monitors convergence, and self-tunes the algorithm's exploration pressure — logging every decision it makes. The project was built to demonstrate how modern agentic AI workflows can be applied to the core challenges of computational biology: the curse of dimensionality, interpretable feature selection, and reproducible analysis pipelines.

---

## Project Overview

| Property | Value |
|---|---|
| Dataset | 50 samples × 5,000 features (synthetic omics) |
| Ground truth | 20 mathematically informative genes embedded in 4,980 noise genes |
| Algorithm | Quantum-Inspired Evolutionary Algorithm (QEA) |
| Fitness metric | Balanced accuracy, 3-fold stratified cross-validation |
| Adaptive controller | `AgentManager` — monitors convergence and tunes Δθ autonomously |
| Outputs | Convergence plot, Manhattan plot, timestamped decision log |

The project is entirely self-contained. One command generates the data, runs the adaptive QEA, evaluates results, and renders two publication-quality plots:

```bash
python main.py
```

### Repository Structure

```
Agentic-Q-Tuner/
├── main.py                   # Single entry point — runs the full pipeline
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py        # Synthetic omics dataset generator
│   ├── q_algorithm.py        # QEA primitives: Q-bits, rotation gate, fitness
│   ├── agent_manager.py      # Adaptive controller (the 'brain')
│   └── visualize_results.py  # Convergence and Manhattan plots
├── agent_logs/               # Timestamped decision logs (auto-created)
└── plots/                    # Output figures (auto-created)
```

---

## Quantum-Inspired Methodology

Classical evolutionary algorithms represent candidate solutions as binary strings and evolve them with crossover and mutation. QEA replaces the binary string with a **quantum register**: each feature *i* is represented by a Q-bit — a pair of real amplitudes (αᵢ, βᵢ) satisfying the normalisation constraint α²ᵢ + β²ᵢ = 1.

### Q-bit Representation

```
Feature i:   [αᵢ, βᵢ]   where   αᵢ² + βᵢ² = 1

P(feature i selected) = βᵢ²
```

Initialising every Q-bit at 45° (αᵢ = βᵢ = 1/√2) gives each feature an equal prior probability of 0.5. The algorithm then tilts these angles over generations based on observed fitness.

### Wavefunction Collapse (Observation)

Each generation, the Q-bit register is *observed*: a binary chromosome is sampled by drawing Bernoulli(βᵢ²) independently per feature. This chromosome — a concrete subset of genes — is then evaluated by a classifier.

### Quantum Rotation Gate

The core update rule. After observing a chromosome *x* and comparing it to the best chromosome *b* found so far, the gate rotates each Q-bit by a signed angle Δθᵢ:

```
| cos(Δθᵢ)  −sin(Δθᵢ) |   | αᵢ |         sign(Δθᵢ) = f(xᵢ, bᵢ, αᵢ·βᵢ)
| sin(Δθᵢ)   cos(Δθᵢ) | × | βᵢ |
```

The sign is determined by the Han & Kim (2002) lookup table: if xᵢ ≠ bᵢ and the current chromosome is inferior, the gate steers βᵢ² toward bᵢ. Features that consistently appear in high-fitness chromosomes accumulate selection probability; noise features drift back toward 0.5 and are gradually eliminated.

### Swap Mutation

To prevent premature convergence, Q-bits are stochastically perturbed by swapping αᵢ ↔ βᵢ (equivalent to a ±π/4 rotation, inverting the current selection probability). This fires at a low per-bit rate (0.5 %) each generation.

**Reference:** Han, K.-H. & Kim, J.-H. (2002). Quantum-inspired evolutionary algorithm for a class of combinatorial optimization. *IEEE Transactions on Evolutionary Computation*, 6(6), 580–593.

---

## Agentic Workflow

The `AgentManager` in `src/agent_manager.py` is the autonomous controller that transforms the QEA from a static algorithm into an adaptive, self-regulating pipeline. It embodies three capabilities that define modern agentic AI systems:

### 1 — Chunked Execution

Rather than running all 60 generations at once, the agent executes the QEA in windows of 5 generations, pausing after each window to assess the state of the search.

### 2 — Fitness Trend Analysis

After every chunk, the agent computes:

```
Δ = global_best_fitness(end of chunk) − global_best_fitness(start of chunk)
```

If Δ < 1 % (absolute), the agent classifies the search as **stalled** — the population has likely converged to a local optimum and requires a perturbation to explore new regions of the 5,000-dimensional feature space.

### 3 — Autonomous Hyper-parameter Adaptation

On detecting a stall, the agent automatically increases the Quantum Rotation Angle:

```
Δθ_new = min(Δθ_current × 1.5,  0.25π)
```

A larger rotation angle moves Q-bits further per generation, forcing broader exploration. When fitness resumes improving, the agent steps the angle back down toward the conservative default (0.05π), switching the search back into exploitation mode. This exploration–exploitation balance is managed entirely without human intervention.

### 4 — Decision Logging

Every decision — and the quantitative reasoning behind it — is persisted to `agent_logs/run_<timestamp>.log`:

```
[2026-04-06 17:14:49] ANALYSIS [Chunk 2] Fitness: 1.0000 → 1.0000  (Δ=+0.0000  threshold=0.01)
[2026-04-06 17:14:49] DECISION [Chunk 2] Fitness stalled at 1.0000 (Δ=+0.0000 < 1%),
                       increasing rotation angle: 0.0500π → 0.0750π  [scale ×1.5]
```

This creates a complete, reproducible audit trail of every action the agent took and why.

---

## Model Iteration & Optimization
Note on Initial Convergence:

During initial testing, the model achieved 100% balanced accuracy within the first 3 generations. However, diagnostic Manhattan Plots revealed that the population had not yet converged on the 20 ground-truth features, instead reaching high accuracy through a redundant combination of noise features.

## How to Run

### Prerequisites

Python 3.11+ is recommended.

```bash
# Clone and enter the repository
git clone <repo-url>
cd Agentic-Q-Tuner

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

This executes all four stages in sequence:

| Stage | Description |
|---|---|
| 1 | Generate the 50 × 5,000 synthetic omics dataset |
| 2 | Run the adaptive QEA under `AgentManager` control (60 generations) |
| 3 | Print results: best accuracy, ground-truth recovery, rotation-angle trace |
| 4 | Save `plots/convergence_plot.png` and `plots/manhattan_plot.png` |

A timestamped decision log is written to `agent_logs/` automatically.

### Run Individual Modules

```bash
# Generate and inspect the dataset only
python src/data_loader.py

# Run the bare QEA without the agent controller
python src/q_algorithm.py

# Regenerate plots from a fresh agent run
python src/visualize_results.py
```

### Output Files

```
plots/
  convergence_plot.png    — Fitness vs. generation; Agent escalation events annotated
  manhattan_plot.png      — β² selection probability for all 5,000 features

agent_logs/
  run_YYYYMMDD_HHMMSS.log — Full timestamped decision log for the run
```

---

## Outputs

### Convergence Plot

Traces the global-best balanced accuracy across all 60 generations. Alternating grey bands mark each 5-generation analysis window. Amber dashed vertical lines mark the exact generations at which the `AgentManager` detected a stall and escalated the Quantum Rotation Angle, with the new Δθ value annotated inline.

### Manhattan Plot

Styled after the GWAS Manhattan plot used in genomics research. Each of the 5,000 features is plotted at its gene index (X-axis) against its mean Q-population selection probability β² (Y-axis). Ground-truth informative genes are highlighted in red; the top-ranked ones are labelled by name. A horizontal dotted line marks the population-mean β² baseline.

---
## Experimental Results & Validation
The Agentic-Q-Tuner was validated against a synthetic high-dimensional dataset ($P=5,000, N=50$) with 20 ground-truth informative features.

1. Autonomous Optimization (Agentic Workflow)
The system demonstrated the necessity of agentic intervention to overcome local optima in high-dimensional search spaces:
-Stall Detection: At Generation 10, the algorithm hit a performance plateau of ~70% balanced accuracy.
-Dynamic Hyperparameter Tuning: The AgentManager identified the stall and successfully increased the Quantum Rotation Angle ($\Delta\theta$) from 0.075π to a maximum of 0.250π.
-Breakthrough Performance: These interventions directly preceded jumps in model fitness, eventually reaching a stabilized accuracy of ~80% without human manual tuning.

2. Feature Discovery & Noise Suppression
The Quantum-Inspired Evolutionary Algorithm (QEA) successfully differentiated biological signals from high-throughput noise:
-Sparsity Enforcement: By implementing a sparsity-aware fitness function, the population mean selection probability was suppressed to 0.434, effectively "silencing" non-contributing features.
-Biomarker Identification: As shown in the Manhattan Plot, the 20 ground-truth informative genes (red) consistently achieved higher selection probabilities compared to the 4,980 noise features.
-Top Hits: Key features such as gene_0016 were successfully prioritized by the Q-population, demonstrating the algorithm's ability to recover sparse signals in a $P \gg N$ context.



## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Vectorised Q-bit arithmetic and RNG |
| `pandas` | Feature-matrix and results management |
| `scikit-learn` | `RandomForestClassifier`, cross-validation, label encoding |
| `matplotlib` | Convergence and Manhattan plots |
