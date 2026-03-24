import itertools
import random

import gymnasium as gym
from dqn import DQN, Replay
import torch
seed = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

episode_over = False
total_reward = 0

# To iteratively decrease exploration
epsilon_start = 1.0
epsilon_decay = 0.9995
epsilon_end = 0.05

class Agent:

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

                s = s_prime
                total_reward += reward

            rewards_episodes.append(reward_episode)
            epsilon_hist.append(epsilon)
            epsilon = max(epsilon_decay * epsilon, epsilon_end)

        env.close()


if __name__ == "__main__":
    agent = Agent()
    agent.run(training=True, render=True)