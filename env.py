import gymnasium as gym
from dqn import DQN
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

episode_over = False
total_reward = 0

class Agent:

    def run(self, training=True, render=False):
        total_reward = 0
        env = gym.make('CartPole-v1', render_mode="human" if render else None)
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        dqn_policy = DQN(n_states, n_actions).to_device(device)

        while not episode_over:
            # Choose an action: 0 = push cart left, 1 = push cart right
            action = env.action_space.sample()  # Random action for now - real agents will be smarter!

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)

            total_reward += reward
            episode_over = terminated or truncated

        env.close()