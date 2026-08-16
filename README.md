# 🐦 Flappy Bird Reinforcement Learning (Deep Q-Network)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-008080.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end **Deep Reinforcement Learning (DRL)** agent trained to master the iconic **Flappy Bird** game using **Deep Q-Networks (DQN)**, **Experience Replay Memory**, and **Target Network Stabilization** in PyTorch and Gymnasium (`flappy-bird-gymnasium`).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Theory](#-architecture--theory)
- [Environment Details](#-environment-details)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
  - [1. Train the Agent](#1-train-the-agent)
  - [2. Evaluate / Watch Trained Agent](#2-evaluate--watch-trained-agent)
  - [3. Play Manually (Human Mode)](#3-play-manually-human-mode)
- [Hyperparameter Configuration](#-hyperparameter-configuration)
- [Training Results & Logs](#-training-results--logs)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## 🌟 Overview

Reinforcement Learning involves an agent interacting with an environment to maximize cumulative future rewards through trial and error. In Flappy Bird, the challenge lies in balancing exploration (trying new actions) and exploitation (using known high-reward actions) while avoiding ground and pipe collisions.

This project implements a Deep Q-Network (DQN) with:
1. **Experience Replay Buffer**: Breaks temporal correlation among consecutive experiences and stabilizes training.
2. **Target Q-Network**: Periodically synchronized with the policy network to prevent divergence in Q-value estimation.
3. **Epsilon-Greedy Exploration with Decay**: Seamless transition from broad random exploration to optimal exploitation.
4. **Hardware Acceleration**: Automatic detection and utilization of **CUDA**, Apple Silicon **MPS**, or CPU.

---

## ⚡ Key Features

- 🧠 **PyTorch Deep Q-Network**: 3-layer feedforward architecture with non-linear ReLU activations.
- 🔄 **Experience Replay**: Custom FIFO replay buffer using double-ended queue (`collections.deque`).
- 🎯 **Target Network Synchronization**: Mitigates non-stationarity in temporal difference updates.
- ⚙️ **Config-Driven Architecture**: Easily tune learning rates, discount factors, batch sizes, and exploration parameters in `parameters.yaml`.
- 🕹️ **Human Gameplay Mode**: Interactive PyGame interface for manual play and testing physics.
- 📊 **Model Checkpointing & Logging**: Automatically saves optimal weights (`.pt`) and performance checkpoints (`.log`).

---

## 📐 Architecture & Theory

### 1. Neural Network Architecture
The DQN network maps the 12-dimensional state vector from `FlappyBird-v0` to 2 Q-values corresponding to the available discrete actions:
- `Action 0`: Do nothing (idle / fall by gravity)
- `Action 1`: Flap wing (propel bird upwards)

```
[ State Input (12) ] ──▶ [ Linear (12 -> 256) ] ──▶ [ ReLU ] ──▶ [ Linear (256 -> 2) ] ──▶ [ Q-Values (Q(s, 0), Q(s, 1)) ]
```

### 2. Bellman Equation & Loss Function
The Bellman Optimality equation is used to compute the target Q-value:

$$Q_{target}(s, a) = r + \gamma \cdot (1 - d) \cdot \max_{a'} Q_{target}(s', a')$$

Where:
- $r$ = Reward received
- $\gamma$ = Discount factor (`gamma: 0.99`)
- $d$ = Termination flag ($1$ if terminal state, $0$ otherwise)
- $s'$ = Next state

The loss function optimized via Adam is the **Mean Squared Error (MSE)** loss:

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( Q(s, a; \theta) - Q_{target}(s, a) \right)^2 \right]$$

---

## 🎮 Environment Details

- **Gymnasium Environment**: `FlappyBird-v0` via [`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium)
- **Observation Space**: 12 numerical features representing:
  - Relative distance to the next top and bottom pipes
  - Bird's vertical position and velocity
  - Next-next pipe positions and gap boundaries
- **Action Space**: Discrete(2) — `0` (Idle), `1` (Flap)
- **Reward Function**:
  - `+0.1` for each frame alive
  - `+1.0` for passing a pipe
  - `-1.0` (or termination penalty) upon collision

---

## 📂 Project Structure

```text
FlappyBird_RL/
├── agent.py                 # Core DQN Agent (training & evaluation loops)
├── dqn.py                   # PyTorch DQN neural network architecture
├── exprience_replay.py      # Replay buffer implementation (Experience Replay)
├── game_flappy_bird.py      # Manual human gameplay script via PyGame
├── parameters.yaml          # Hyperparameters configuration
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore patterns
└── README.md                # Project documentation
```

---

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/PrimeStudent1/FlappyBird_RL.git
cd FlappyBird_RL
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Agent
To start training the DQN agent using the `flappybirdv0` parameter profile:
```bash
python agent.py flappybirdv0 --train
```

*During training, the agent prints episode rewards, exploration rates (epsilon), and periodically saves the best model weights to `runs/flappybirdv0.pt`.*

### 2. Evaluate / Watch Trained Agent
To load the trained weights and watch the AI play in real-time with visual rendering:
```bash
python agent.py flappybirdv0
```

### 3. Play Manually (Human Mode)
To play the game yourself using the spacebar:
```bash
python game_flappy_bird.py
```
- **Controls**: Press <kbd>Space</kbd> to flap. Close window to exit.

---

## ⚙️ Hyperparameter Configuration

Hyperparameters are decoupled in `parameters.yaml` for clean experimentation:

```yaml
flappybirdv0:
  env_id: FlappyBird-v0
  epsilon_init: 1.0           # Initial exploration rate
  epsilon_min: 0.05           # Minimum exploration floor
  epsilon_decay: 0.9995       # Multiplicative epsilon decay per episode
  replay_memory_size: 100000  # Max capacity of experience replay buffer
  mini_batch_size: 32         # Mini-batch size for gradient updates
  network_sync_rate: 10       # Target network weight sync interval (steps)
  alpha: 0.001                # Adam optimizer learning rate
  gamma: 0.99                 # Discount factor for future rewards
  reward_threshold: 1000      # Episode cutoff threshold
```

---

## 📈 Training Results & Logs

During training:
- Checkpoints are saved under the `runs/` directory.
- `runs/flappybirdv0.pt`: Best policy neural network weights.
- `runs/flappybirdv0.log`: Log of best achieved episode rewards and progression.

---

## 🔮 Future Roadmap

- [ ] **Double DQN (DDQN)**: Decouple action selection from evaluation to mitigate overestimation bias.
- [ ] **Dueling DQN**: Separate state-value $V(s)$ and advantage $A(s, a)$ streams.
- [ ] **Prioritized Experience Replay (PER)**: Sample transitions proportional to their TD error.
- [ ] **Vision-based Input (CNN)**: Train an agent directly on raw game screen pixels.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

⭐ If you found this project helpful, please consider giving it a star!
