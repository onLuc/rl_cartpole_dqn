import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed; saves PNGs directly
from matplotlib import pyplot as plt
from env import Agent

# ── Constants ──────────────────────────────────────────────────────────────────
N_REPS          = 5
ABLATION_STEPS  = 1_000_000
FINAL_STEPS     = 1_000_000
RESULTS_DIR     = "results_final_v3"
SMOOTH_WINDOW   = 200   # rolling average window in episodes (higher = smoother)
LINEAR_DECAY    = True  # True = linear decay over steps, False = multiplicative decay per episode
DECAY_TAG       = "linear" if LINEAR_DECAY else "nonlinear"  # used in cache filenames
SEEDS           = list(range(N_REPS))   # reps use seeds 0,1,2,3,4 → reproducible

os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline hyperparameters (fixed while varying one at a time in ablation)
BASELINE = dict(
    lr            = 1e-3,
    update_freq   = 16,
    hidden_size   = 128,
    epsilon_decay = 0.99,   # used when LINEAR_DECAY=False, every 1000 timesteps
    epsilon_end   = 0.05,    # used when LINEAR_DECAY=True (also floor for nonlinear)
    use_er        = True,
    use_tn        = True,
)

# ── Ablation grids (Task 2.2) ───────────────────────────────────────────────────
# The exploration parameter differs by decay mode:
#   linear    → ablate epsilon_end  (final exploration rate, decay rate is fixed)
#   nonlinear → ablate epsilon_decay (controls how fast exploration falls off)
ABLATION = {
    "lr":          [1e-4,  1e-3,  5e-3],
    "update_freq": [1, 4, 16, 64],
    "hidden_size": [32, 64, 128, 256],
}
if LINEAR_DECAY:
    ABLATION["epsilon_end"]   = [0.01, 0.05, 0.2]
else:
    ABLATION["epsilon_decay"] = [0.95, 0.99, 0.999]

# ── 4-configuration comparison (Task 2.4) ──────────────────────────────────────
CONFIGS = {
    "Naive":   dict(use_er=False, use_tn=False),
    "Only TN": dict(use_er=False, use_tn=True),
    "Only ER": dict(use_er=True,  use_tn=False),
    "TN & ER": dict(use_er=True,  use_tn=True),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def smooth(values, window=50):
    out = []
    for i in range(len(values)):
        out.append(np.mean(values[max(0, i - window + 1):i + 1]))
    return np.array(out)


def run_experiment(run_kwargs, max_steps, label=""):
    """Run N_REPS repetitions with fixed seeds. Returns lists of arrays."""
    all_rewards, all_steps = [], []
    hp_str = "  ".join(f"{k}={v}" for k, v in run_kwargs.items())
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {hp_str}")
    print(f"  max_steps={max_steps:,}  reps={N_REPS}")
    print(f"{'='*60}")
    for rep, seed in enumerate(SEEDS):
        print(f"  rep {rep + 1}/{N_REPS}  (seed={seed})")
        agent = Agent()
        agent.run(training=True, render=False,
                  max_steps=max_steps, run_seed=seed,
                  linear_decay=LINEAR_DECAY, **run_kwargs)
        all_rewards.append(np.array(agent.rewards_episodes))
        all_steps.append(np.array(agent.steps_per_episode))
    return all_rewards, all_steps


def cache_path(name):
    return os.path.join(RESULTS_DIR, f"{name}.npz")


def save_results(name, all_rewards, all_steps):
    np.savez(cache_path(name),
             rewards=np.array(all_rewards, dtype=object),
             steps=np.array(all_steps,   dtype=object))
    print(f"  Saved → {cache_path(name)}")


def load_results(name):
    data = np.load(cache_path(name), allow_pickle=True)
    rewards = [np.array(r, dtype=np.float64) for r in data["rewards"]]
    steps   = [np.array(s, dtype=np.float64) for s in data["steps"]]
    return rewards, steps


def load_or_run(name, run_kwargs, max_steps, label):
    tagged = f"{DECAY_TAG}_{name}"
    if os.path.exists(cache_path(tagged)):
        print(f"  Loading cached: {tagged}")
        return load_results(tagged)
    print(f"\nRunning: {label} ({max_steps:,} steps × {N_REPS} reps)")
    rewards, steps = run_experiment(run_kwargs, max_steps, label=label)
    save_results(tagged, rewards, steps)
    return rewards, steps


def to_step_grid(all_rewards, all_steps, n_points=500):
    """Interpolate each rep onto a shared step grid, then average."""
    max_s = max(s[-1] for s in all_steps)
    grid = np.linspace(0, max_s, n_points)
    interp = [np.interp(grid, steps, smooth(rewards, window=SMOOTH_WINDOW))
              for rewards, steps in zip(all_rewards, all_steps)]
    arr = np.array(interp)
    return grid, arr.mean(0), arr.std(0)


def save_fig(fig, filename):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {path}")
    plt.close(fig)


# ── Task 2.1: Basic training curve ────────────────────────────────────────────

def run_basic():
    rewards, steps = load_or_run("basic_training", BASELINE, FINAL_STEPS, "Basic")

    grid, mean, std = to_step_grid(rewards, steps)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Baseline DQN on CartPole (5 seeds)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(f"Return (smoothed over {SMOOTH_WINDOW} ep)")
    ax.plot(grid, mean, color="steelblue", label="Mean")
    ax.fill_between(grid, mean - std, mean + std, alpha=0.3, color="steelblue", label="±1 std")
    ax.axhline(500, color="green", linestyle="--", label="Optimum (500)")
    ax.set_ylim(0, 520)
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "basic_training.png")


# ── Task 2.2: Ablation study ───────────────────────────────────────────────────

def run_ablation():
    for param, values in ABLATION.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"Ablation: {param}  ({ABLATION_STEPS:,} steps)")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(f"Return (smoothed over {SMOOTH_WINDOW} ep)")

        for val in values:
            kwargs = {**BASELINE, param: val}
            name   = f"ablation_{param}_{val}"
            label  = f"{param}={val}"
            rewards, steps = load_or_run(name, kwargs, ABLATION_STEPS, label)
            grid, mean, std = to_step_grid(rewards, steps)
            ax.plot(grid, mean, label=label)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2)

        ax.set_ylim(0, 520)
        ax.legend()
        plt.tight_layout()
        save_fig(fig, f"ablation_{param}.png")


# ── Task 2.4: Configuration comparison ────────────────────────────────────────

def run_config_comparison():
    # Use the baseline hyperparameters (update after ablation if desired)
    hp = {k: BASELINE[k] for k in ("lr", "update_freq", "hidden_size", "epsilon_decay", "epsilon_end")}

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Naive  /  Only TN  /  Only ER  /  TN & ER")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(f"Return (smoothed over {SMOOTH_WINDOW} ep)")

    for config_name, flags in CONFIGS.items():
        kwargs = {**hp, **flags}
        name   = f"config_{config_name.replace(' ', '_')}"
        rewards, steps = load_or_run(name, kwargs, FINAL_STEPS, config_name)
        grid, mean, std = to_step_grid(rewards, steps)
        ax.plot(grid, mean, label=config_name)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15)

    ax.set_ylim(0, 520)
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "config_comparison.png")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["basic", "ablation", "configs", "all"],
                        default="all")
    args = parser.parse_args()

    if args.task in ("ablation", "all"):
        run_ablation()
    if args.task in ("basic", "all"):
        run_basic()
    if args.task in ("configs", "all"):
        run_config_comparison()
