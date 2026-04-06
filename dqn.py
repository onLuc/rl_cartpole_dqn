import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
torch.random.manual_seed(1)


class DQN(nn.Module):
    def __init__(self, n_states, n_actions, hidden=512):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(n_states, hidden)
        self.output_layer = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        return self.output_layer(x)

class Replay:
    def __init__(self, max_size, seed):
        self.mem = deque([], maxlen=max_size) # fifo
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.mem.append(transition)

    def sample(self, sample_size):
        return random.sample(self.mem, sample_size)

    def __len__(self):
        return len(self.mem)

if __name__ == "__main__":
    n_states = 100
    n_actions = 2
    nn = DQN(n_states, n_actions)
    out = nn(torch.randn(1, n_states))
    print(out)
