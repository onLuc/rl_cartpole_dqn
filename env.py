import itertools
import random

import gymnasium as gym
from dqn import DQN, Replay
import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
seed = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

episode_over = False
total_reward = 0

# Default exploration schedule
epsilon_start = 1.0
epsilon_end = 0.05


class Agent:
    def __init__(self):
        self.gamma = .99
        self.loss_function = nn.MSELoss()
        self.optimizer = None

    def run(self, training=True, render=False, max_steps=1_000_000,
            use_er=True, use_tn=True,
            lr=1e-3, update_freq=1, hidden_size=512, epsilon_decay=0.9995):
        """
        use_er:       use experience replay buffer
        use_tn:       use separate target network
        lr:           learning rate
        update_freq:  optimize every N environment steps
        hidden_size:  number of hidden units in DQN
        epsilon_decay: multiplicative decay per episode
        """
        total_reward = 0
        env = gym.make('CartPole-v1', render_mode="human" if render else None)
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n

        dqn_policy = DQN(n_states, n_actions, hidden=hidden_size).to(device)
        self.rewards_episodes = []
        self.epsilon_hist = []
        self.steps_per_episode = []  # cumulative steps at end of each episode

        if training:
            replay_memory = Replay(10000, seed) if use_er else None
            epsilon = epsilon_start

            dqn_target = DQN(n_states, n_actions, hidden=hidden_size).to(device)
            if use_tn:
                dqn_target.load_state_dict(dqn_policy.state_dict())

            step_count = 0   # steps since last target sync
            total_steps = 0

            self.optimizer = torch.optim.Adam(dqn_policy.parameters(), lr=lr)

        for episode in itertools.count():
            s, _ = env.reset()
            s = torch.tensor(s, dtype=torch.float, device=device)
            terminated = False
            reward_episode = 0

            while not terminated:
                if training and random.random() < epsilon:
                    a = env.action_space.sample()
                    a = torch.tensor(a, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        a = dqn_policy.forward(s.unsqueeze(0)).squeeze().argmax()

                s_prime, reward, terminated, truncated, info = env.step(int(a.item()))
                reward_episode += reward

                s_prime = torch.tensor(s_prime, dtype=torch.float, device=device)
                reward_t = torch.tensor(reward, dtype=torch.float, device=device)

                if training:
                    total_steps += 1
                    step_count += 1

                    if use_er:
                        replay_memory.append((s, a, s_prime, reward_t, terminated))
                        if len(replay_memory) > 32 and total_steps % update_freq == 0:
                            mini_batch = replay_memory.sample(32)
                            target_net = dqn_target if use_tn else dqn_policy
                            self.optimize(mini_batch, dqn_policy, target_net)
                    else:
                        # No ER: train on single transition
                        if total_steps % update_freq == 0:
                            mini_batch = [(s, a, s_prime, reward_t, terminated)]
                            target_net = dqn_target if use_tn else dqn_policy
                            self.optimize(mini_batch, dqn_policy, target_net)

                    if use_tn and step_count >= 10:
                        dqn_target.load_state_dict(dqn_policy.state_dict())
                        step_count = 0

                s = s_prime
                total_reward += reward_t

            self.rewards_episodes.append(reward_episode)
            self.epsilon_hist.append(epsilon)
            self.steps_per_episode.append(total_steps)
            epsilon = max(epsilon_decay * epsilon, epsilon_end)

            if episode % 50 == 0:
                avg = np.mean(self.rewards_episodes[-50:])
                print(f"Episode {episode:>5} | Steps {total_steps:>8} | Return {reward_episode:>6.1f} | Avg(50) {avg:>6.1f} | Epsilon {epsilon:.3f}")

            if training and total_steps >= max_steps:
                break

        env.close()

    def optimize(self, mini_batch, dqn_policy, dqn_target):
        states, actions, states_p, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions).long().to(device)
        states_p = torch.stack(states_p)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            Q_target = rewards + (1 - terminations) * self.gamma * dqn_target(states_p).max(1)[0]
        Q_current = dqn_policy(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_function(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Reward per Episode")
        plt.plot(self.rewards_episodes, label='Raw Reward', alpha=0.3)
        if len(self.rewards_episodes) >= 25:
            means = [np.mean(self.rewards_episodes[max(0, i - 25):i + 1]) for i in range(len(self.rewards_episodes))]
            plt.plot(means, label='Average (25 ep)', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Epsilon Decay")
        plt.plot(self.epsilon_hist, color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    agent = Agent()
    agent.run(training=True, render=False)
