"""
Deep Q-Learning (DQN) Agent for Flappy Bird Gymnasium Environment.

This module implements a DQN agent using PyTorch, featuring:
- Experience Replay Memory buffer for experience sampling
- Target Q-Network stabilization with periodic weight updates
- Epsilon-greedy policy with exponential decay for exploration vs exploitation
- Vectorized batch optimization for fast gradient updates
- Hyperparameter loading from external YAML configuration
"""

import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from exprience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import argparse

# Hardware acceleration setup (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

runs_dir = "runs"
os.makedirs(runs_dir, exist_ok=True)


class Agent:
    """
    DQN Agent capable of training and evaluating on the FlappyBird environment.
    """
    def __init__(self, param_set):
        self.param_set = param_set
        with open("parameters.yaml", "r") as f:
            all_params_set = yaml.safe_load(f)
            params = all_params_set[param_set]

        # Learning parameters
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        # Epsilon-greedy exploration parameters
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        # Replay memory & batch size
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        # Target network sync & convergence criteria
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Checkpoints and logs
        self.log_file = os.path.join(runs_dir, f"{self.param_set}.log")
        self.model_file = os.path.join(runs_dir, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        """
        Execute training or evaluation loop for FlappyBird.
        """
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]  # Input state dimension
        num_actions = env.action_space.n             # Output action choices

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # Synchronize weights initially from policy network to target network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while not terminated and episode_reward < self.reward_threshold:
                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # Explore: random action
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        # Exploit: best action according to policy network
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Step through environment
                next_state, reward, terminated, _, _ = env.step(action.item())

                # Convert outputs to PyTorch Tensors
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state
                episode_reward += reward.item()

            print(f"Episode {episode + 1} | Total Reward: {episode_reward} | Epsilon: {epsilon:.4f}")

            if is_training:
                # Decay exploration rate (epsilon)
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save model checkpoint on new best episode reward
                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward} for episode = {episode + 1}"
                    with open(self.log_file, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.model_file)
                    best_reward = episode_reward
            else:
                # Load pre-trained best policy model for evaluation
                policy_dqn.load_state_dict(torch.load(self.model_file))
                policy_dqn.eval()

            # Train network on mini-batch from replay memory
            if is_training and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Sync target network weights periodically
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Perform vectorized batch gradient descent update on policy DQN.
        """
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Calculate target Q-values using Bellman equation: Q_target = reward + gamma * max(Q_target(s'))
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # Calculate predicted Q-values from current policy for chosen actions
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute Mean Squared Error Loss
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation & weight update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Command line argument parser
    # Usage:
    #   Training: python agent.py flappybirdv0 --train
    #   Testing:  python agent.py flappybirdv0
    parser = argparse.ArgumentParser(description='Train or test FlappyBird RL DQN agent.')
    parser.add_argument('hyperparameters', help='Name of hyperparameter set in parameters.yaml')
    parser.add_argument('--train', help='Train the agent', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)