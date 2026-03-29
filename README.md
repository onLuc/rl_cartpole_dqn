# Assignment 2: Q-Learning with Deep Function Approximation (DQN) on CartPole

Implementation of Deep Q-Learning (DQN) on the CartPole-v1 environment, including a target network, experience replay buffer, and a full ablation study over key hyperparameters. Best hyperparameters were determined via ablation and are used for all final experiments.

---

## Requirements

- Python 3.12
- CUDA-capable GPU (optional but recommended)

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` was generated with `python -m pip freeze` and pins all exact versions used during the experiments.

---

## Project Structure

```
.
├── dqn.py            # DQN network architecture and Replay buffer
├── env.py            # Agent class with training loop
├── experiment.py     # Experiment runner (ablation, basic, config comparison)
├── requirements.txt
├── results/          # Auto-generated plots and cached .npz result files
└── BaselineDataCartPole.csv  # Provided baseline for reference
```

---

## Reproducing Experiments

All experiments are run through `experiment.py`. Results are **cached** to `results/*.npz` after each run — re-running a task loads from cache and regenerates plots without retraining.

### Key settings (top of `experiment.py`)

| Constant | Value | Description |
|---|---|---|
| `N_REPS` | `5` | Repetitions per configuration |
| `SEEDS` | `[0,1,2,3,4]` | Fixed per-rep seeds for full reproducibility |
| `ABLATION_STEPS` | `500,000` | Environment steps per ablation run |
| `FINAL_STEPS` | `1,000,000` | Environment steps for basic and config runs |
| `SMOOTH_WINDOW` | `200` | Rolling average window (episodes) used in all plots |
| `LINEAR_DECAY` | `True` | Epsilon decay mode — see section below |

---

## Epsilon Decay Modes

The exploration schedule can be switched with the `LINEAR_DECAY` flag in `experiment.py`.

### Linear decay (`LINEAR_DECAY = True`) — **default, recommended**
Epsilon decreases linearly from `epsilon_start=1.0` to `epsilon_end` over the total training steps:

```
epsilon = max(1.0 - (1.0 - epsilon_end) * (step / max_steps), epsilon_end)
```

All seeds have **identical epsilon at every step count**, which eliminates cross-seed exploration variance when plotting return vs environment steps. This produces significantly cleaner learning curves and tighter confidence bands. The ablation parameter for this mode is `epsilon_end`.

### Multiplicative decay (`LINEAR_DECAY = False`)
Epsilon is multiplied by `epsilon_decay` after each episode:

```
epsilon = max(epsilon * epsilon_decay, epsilon_end)
```

Episode lengths vary between seeds, so at any given step count different seeds may be at different epsilon values. This introduces additional variance into the learning curves. The ablation parameter for this mode is `epsilon_decay`.

> Cache files are automatically tagged `linear_*.npz` or `nonlinear_*.npz` so results from both modes can coexist without conflict.

---

## Baseline Hyperparameters

Determined by ablation study (Task 2.2). Bold values in the ablation tables indicate the winning value used as baseline.

| Hyperparameter | Value | Description |
|---|---|---|
| `lr` | **1e-3** | Adam learning rate |
| `update_freq` | **16** | Optimize every N environment steps |
| `hidden_size` | **64** | Hidden units in the DQN |
| `epsilon_end` | **0.05** | Final exploration rate (linear mode) |
| `epsilon_decay` | **0.999** | Per-episode decay factor (nonlinear mode) |
| `epsilon_start` | 1.0 | Initial exploration rate (fixed) |
| `use_er` | True | Experience replay enabled |
| `use_tn` | True | Target network enabled |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 32 | Replay buffer sample size |
| `replay_buffer` | 10,000 | Maximum replay buffer capacity |

---

## Running Experiments

### Task 2.1 — Basic learning curve

Trains the full DQN (TN + ER) with the best baseline hyperparameters. Plots return over environment steps across 5 seeds with ±1 std shading.

```bash
python experiment.py --task basic
```

Output: `results/linear_basic_training.npz`, `results/basic_training.png`

---

### Task 2.2 — Ablation study

Varies one hyperparameter at a time (3 values each) while keeping all others at baseline. Produces one plot per hyperparameter.

```bash
python experiment.py --task ablation
```

Output: `results/ablation_lr.png`, `results/ablation_update_freq.png`, `results/ablation_hidden_size.png`, `results/ablation_epsilon_end.png` (linear mode)

**Ablation grids:**

**Linear decay mode (`LINEAR_DECAY = True`)**

| Parameter | Small | Medium | Large | Baseline |
|---|---|---|---|---|
| Learning rate | 1e-4 | **1e-3** | 1e-2 | **1e-3** |
| Update frequency | 1 | 4 | **16** | **16** |
| Network size | **64** | 256 | 512 | **64** |
| Final epsilon (`epsilon_end`) | 0.01 | **0.05** | 0.2 | **0.05** |

**Nonlinear decay mode (`LINEAR_DECAY = False`)**

| Parameter | Small | Medium | Large | Baseline |
|---|---|---|---|---|
| Learning rate | 1e-4 | **1e-3** | 1e-2 | **1e-3** |
| Update frequency | 1 | 4 | **16** | **16** |
| Network size | **64** | 256 | 512 | **64** |
| Epsilon decay (`epsilon_decay`) | **0.999** | 0.9995 | 0.9999 | **0.999** |

---

### Task 2.4 — Configuration comparison

Compares all four DQN configurations in a single plot using the baseline hyperparameters:

| Configuration | Target Network | Experience Replay |
|---|---|---|
| Naive | No | No |
| Only TN | Yes | No |
| Only ER | No | Yes |
| TN & ER | Yes | Yes |

```bash
python experiment.py --task configs
```

Output: `results/linear_config_*.npz`, `results/config_comparison.png`

---

### Run everything

```bash
python experiment.py --task all
```

Runs ablation → basic → configs in order. Cached results are loaded automatically.

---

## Notes

- Plots are saved as PNGs to `results/` — no display required, fully headless.
- Each run prints per-episode logs: episode, total steps, return, rolling average (50 ep), epsilon.
- To force a re-run of a cached result, delete the corresponding `.npz` file in `results/`.
- All seeds, hyperparameters, and decay mode are fixed in `experiment.py` — no arguments needed for full reproducibility.
