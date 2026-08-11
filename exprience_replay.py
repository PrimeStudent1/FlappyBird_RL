from collections import deque
import random

class ReplayMemory:
    """
    Experience Replay Buffer using a FIFO queue (deque).
    Stores past agent transitions (state, action, next_state, reward, done)
    to break temporal correlation during DQN training.
    """
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, new_exp):
        """Add a transition tuple to memory."""
        self.memory.append(new_exp)

    def sample(self, sample_size):
        """Randomly sample a mini-batch of transitions from memory."""
        return random.sample(self.memory, sample_size)

    def __len__(self):
        """Return the current number of saved experiences."""
        return len(self.memory)