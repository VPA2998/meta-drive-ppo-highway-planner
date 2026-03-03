# 🤖 MetaDrive PPO Highway Planner

**End-to-end highway motion planning using Proximal Policy Optimization (PPO) in the MetaDrive simulator.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![RL](https://img.shields.io/badge/RL-PPO-orange)

> 🏗️ See [System Architecture](docs/ARCHITECTURE.md) for MDP formulation, network architecture, and training pipeline.

---

## 📋 Overview

This project implements **end-to-end autonomous driving** using Deep Reinforcement Learning:

- **Environment**: MetaDrive (procedurally generated highway scenarios)
- **Algorithm**: PPO (Proximal Policy Optimization) Actor-Critic
- **Observation Space**: LiDAR + Side Detectors + Lane Line (261-dim vector)
- **Action Space**: Continuous steering + throttle/brake (2-dim)
- **Training**: 100k+ timesteps on 50 diverse scenarios
- **Evaluation**: Generalization test on **unseen seeds** (1000+)
- **Deployment**: Gradio interactive demo with pre-generated rollouts

**Key Achievement**: The trained policy achieves **87+ mean reward** on completely unseen highway configurations, demonstrating strong generalization.

---

## 🎯 Key Features

### PPO Agent Architecture
- **Policy Network**: MlpPolicy `[256, 256]` hidden layers
- **Value Network**: `[256, 256]` hidden layers
- **Activation**: ReLU
- **Learning Rate**: 3e-4
- **Entropy Coefficient**: 0.01 (exploration bonus)

### Training Highlights
✅ **Generalization**: Trained on seeds 0–50, evaluated on seeds 1000+  
✅ **Reward Shaping**: Survival bonus + distance traveled + collision penalties  
✅ **Domain Randomization**: Random lane width, vehicle dynamics, traffic density  
✅ **Monitoring**: Real-time logs (episodic reward, rollout stats, FPS)

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Total Timesteps | 100,000 |
| Mean Reward (Train) | 100.5 |
| Mean Reward (Eval) | 87.2 ± 30.8 |
| Success Rate | ~80% |
| Generalization Gap | <15% |

---

## 📁 Project Structure

```
meta-drive-ppo-highway-planner/
├── notebooks/
│ └── 01_ppo_metadrive_training.ipynb # Full pipeline
├── src/ # Modular packages (future)
├── models/ # Saved PPO policies (.zip)
├── outputs/ # GIFs, videos
│ └── demo/ # Demo assets
├── results/ # Training logs, CSVs
├── docs/ # Architecture docs
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/VPA2998/meta-drive-ppo-highway-planner.git
cd meta-drive-ppo-highway-planner
python3 -m venv .venv
source .venv/bin/activate  # WSL/Linux
pip install -r requirements.txt
```
### 2. Run the Notebook
```bash
jupyter notebook
```

Open [notebooks/01_ppo_metadrive_training.ipynb](notebooks/01_ppo_metadrive_training.ipynb.ipynb) and run all cells.

**What happens:**

- ✅ Initializes MetaDrive env (LiDAR + detectors)

- ✅ Configures PPO agent (MlpPolicy 256x256)

- ✅ Trains for 100k timesteps (~10-15 min on GPU)

- ✅ Evaluates on unseen scenarios (seed 1000+)

- ✅ Generates 9 GIFs (3 seeds × 3 traffic densities)

- ✅ Launches Gradio demo for visualization

### 🎬 Demo

(Coming soon: Add GIF after running notebook Section 5)

*PPO agent navigating highway with varying traffic densities (0.05, 0.1, 0.2). Policy generalizes to unseen seeds.*

### 📊 Training Results

From the latest run (100k steps):


| rollout | eplenmean 283 | eprewmean 100 |
|----------|----------------|---------------|
| train | approxkl 0.0059 | clipfraction 0.098 |
|        | entropyloss -2.25 | explainedvariance 0.239 |
|        | loss 25.3 | policygradientloss 0.001 |
|        | valueloss 45.3 | std 0.744 |

**Key Observations:**

- Episodic reward steadily increased from ~4 → 100

- Policy entropy decreased naturally (0.98 → 0.74)

- Value loss stabilized after 60k steps

- No catastrophic forgetting on unseen seeds


