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

episode_over = False
total_reward = 0

# To iteratively decrease exploration
epsilon_start = 1.0
epsilon_decay = 0.9995
epsilon_end = 0.05

class Agent:
    def __init__(self):
        self.gamma = .99 # discount factor
        self.loss_function = nn.MSELoss()
        self.optimizer = None

    def run(self, training=True, render=False):
        total_reward = 0
        env = gym.make('CartPole-v1', render_mode="human" if render else None)
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n

        dqn_policy = DQN(n_states, n_actions).to(device)
        rewards_episodes = []
        epsilon_hist = []

        if training:
            replay_memory = Replay(10000, seed)
            epsilon = epsilon_start
            dqn_target = DQN(n_states, n_actions).to(device)
            dqn_target.load_state_dict(dqn_policy.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(dqn_policy.parameters(), lr=1e-3)

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
                    # Save cycles, since we're not training
                    with torch.no_grad():
                        a = dqn_policy.forward(s.unsqueeze(0)).squeeze().argmax()
                # Choose an action: 0 = push cart left, 1 = push cart right

                # Take the action and see what happens
                s_prime, reward, terminated, truncated, info = env.step(int(a.item()))
                reward_episode += reward

                s_prime = torch.tensor(s_prime, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                # reward: +1 for each step the pole stays upright
                # terminated: True if pole falls too far (agent failed)
                # truncated: True if we hit the time limit (500 steps)
                if training:
                    replay_memory.append((s, a, s_prime, reward, terminated))
                    step_count += 1

                s = s_prime
                total_reward += reward

            rewards_episodes.append(reward_episode)
            epsilon_hist.append(epsilon)
            epsilon = max(epsilon_decay * epsilon, epsilon_end)

            if len(replay_memory) > 32:
                mini_batch = replay_memory.sample(32)
                self.optimize(mini_batch, dqn_policy, dqn_target)

                if step_count > 10:
                    dqn_target.load_state_dict(dqn_policy.state_dict())
                    step_count = 0

            # print(len(rewards_episodes))
            # if len(rewards_episodes) > 100:
            #     break

        env.close()

    def optimize(self, mini_batch, dqn_policy, dqn_target):
        states, actions, states_p, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions).long().to(device)
        states_p = torch.stack(states_p)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            Q_target = rewards + (1-terminations) * self.gamma * dqn_target(states_p).max(1)[0]
        Q_current = dqn_policy(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_function(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot(self):
        # if not hasattr(self, 'rewards_episodes') or not self.rewards_episodes:
        #     print("No data to plot. Run the agent first.")
        #     return

        plt.figure(figsize=(12, 5))

        # --- Plot 1: Rewards ---
        plt.subplot(1, 2, 1)
        plt.title("Reward per Episode")
        plt.plot(self.rewards_episodes, label='Raw Reward', alpha=0.3)

        # Calculate running average (window of 25 episodes)
        if len(self.rewards_episodes) >= 25:
            means = [np.mean(self.rewards_episodes[max(0, i - 25):i + 1]) for i in range(len(self.rewards_episodes))]
            plt.plot(means, label='Average (25 ep)', color='red')

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        # --- Plot 2: Epsilon Decay ---
        plt.subplot(1, 2, 2)
        plt.title("Epsilon Decay")
        plt.plot(self.epsilon_hist, color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    agent = Agent()
    agent.run(training=True, render=True)
    # agent.plot()