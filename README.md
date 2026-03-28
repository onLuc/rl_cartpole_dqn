# Assignment 2: Q-Learning with Deep Function Approximation (DQN) on CartPole

Implementation of Deep Q-Learning (DQN) on the CartPole-v1 environment, including a target network, experience replay, and a full ablation study over key hyperparameters.

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

All experiments are run through `experiment.py`. Results are **cached** to `results/*.npz` after each run, so individual tasks can be re-run or re-plotted without retraining.

### Key settings (top of `experiment.py`)

| Constant | Value | Description |
|---|---|---|
| `N_REPS` | 5 | Repetitions per configuration |
| `SEEDS` | `[0,1,2,3,4]` | Fixed seeds for reproducibility |
| `ABLATION_STEPS` | 500,000 | Steps per ablation run |
| `FINAL_STEPS` | 1,000,000 | Steps for basic and config runs |
| `SMOOTH_WINDOW` | 200 | Episode window for rolling average in plots |

---

### Task 2.1 — Basic learning curve

Trains DQN with Target Network + Experience Replay using the baseline hyperparameters. Plots return over environment steps across 5 seeds.

```bash
python experiment.py --task basic
```

Output: `results/basic_training.png`

---

### Task 2.2 — Ablation study

Varies one hyperparameter at a time (3 values each) while keeping all others fixed at baseline. Produces one plot per hyperparameter.

```bash
python experiment.py --task ablation
```

Output: `results/ablation_lr.png`, `results/ablation_update_freq.png`, `results/ablation_hidden_size.png`, `results/ablation_epsilon_decay.png`

Hyperparameters ablated:

| Parameter | Values tested | Baseline |
|---|---|---|
| Learning rate (`lr`) | 1e-4, **1e-3**, 1e-2 | 1e-3 |
| Update frequency (`update_freq`) | 1, 4, **16** | 16 |
| Network size (`hidden_size`) | **64**, 256, 512 | 64 |
| Epsilon decay (`epsilon_decay`) | **0.999**, 0.9995, 0.9999 | 0.999 |

---

### Task 2.4 — Configuration comparison

Compares all four DQN configurations on a single plot, using the baseline hyperparameters:

- **Naive** — no target network, no experience replay
- **Only TN** — target network only
- **Only ER** — experience replay only
- **TN & ER** — full DQN (both)

```bash
python experiment.py --task configs
```

Output: `results/config_comparison.png`

---

### Run everything

```bash
python experiment.py --task all
```

Runs ablation → basic → configs in order. Already-cached results are skipped automatically.

---

## Baseline hyperparameters

```python
lr            = 1e-3
update_freq   = 16       # optimize every 16 environment steps
hidden_size   = 64       # hidden units in the DQN
epsilon_decay = 0.999    # multiplicative decay per episode
use_er        = True
use_tn        = True
gamma         = 0.99
batch_size    = 32
replay_buffer = 10,000
```

---

## Notes

- Plots are saved to `results/` as PNG files (no display required, runs headless).
- Each run prints per-episode logs: episode number, total steps, return, rolling average (50 ep), and current epsilon.
- To force a re-run of a cached experiment, delete the corresponding `.npz` file in `results/`.
