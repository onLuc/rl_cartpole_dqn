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

## Running Experiments



### Run everything

```bash
python experiment.py --task all
```

Runs ablation → basic → configs in order. Cached results are loaded automatically.

---

### Basic learning curve

Trains the full DQN (TN + ER) with the best baseline hyperparameters. Plots return over environment steps across 5 seeds with ±1 std shading.

```bash
python experiment.py --task basic
```

Output: `results/linear_basic_training.npz`, `results/basic_training.png`

---

### Ablation study

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

### Configuration comparison

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


## Notes

- Plots are saved as PNGs to `results/` — no display required, fully headless.
- Each run prints per-episode logs: episode, total steps, return, rolling average (50 ep), epsilon.
- To force a re-run of a cached result, delete the corresponding `.npz` file in `results/`.
- All seeds, hyperparameters, and decay mode are fixed in `experiment.py` — no arguments needed for full reproducibility.
