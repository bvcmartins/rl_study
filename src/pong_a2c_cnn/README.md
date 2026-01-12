# Pong A2C CNN v1

Advantage Actor-Critic (A2C) implementation for Atari Pong, building on the optimized architecture from Double DQN.

## Overview

This implementation expands on the `pong_double_dqn_cnn` codebase by transitioning from value-based (DQN) to policy-based (A2C) reinforcement learning.

### Key Differences from Double DQN

| Aspect | Double DQN | A2C (This Implementation) |
|--------|------------|---------------------------|
| **Learning Type** | Value-based (Q-learning) | Policy-based (Policy Gradient) |
| **Policy** | Implicit (argmax Q) | Explicit stochastic policy |
| **Exploration** | ε-greedy | Entropy bonus |
| **Training** | Off-policy with replay buffer | On-policy with trajectory rollouts |
| **Network Output** | Q-values for each action | Policy distribution + State value |
| **Update Rule** | Bellman equation | Policy gradient + Value function |
| **Target Network** | Required | Not needed |

## Architecture

### Network Structure

```
Input: 2×84×84 stacked frames
    ↓
Shared CNN Feature Extractor (32→36→20 filters)
    ├─────────┬─────────┤
    ↓         ↓         ↓
Features   Actor   Critic
           Head     Head
            ↓         ↓
         Policy    Value
```

### Components

1. **Shared CNN** (`networks.py::SharedCNNFeatures`)
   - Reuses optimized 32→36→20 filter architecture from Double DQN
   - Extracts spatial features from stacked frames
   - Shared between actor and critic for efficiency

2. **Actor** (`networks.py::Actor`)
   - Maps features → 512 → action_size
   - Outputs policy logits (converted to probabilities via softmax)
   - Learns which actions to take in each state

3. **Critic** (`networks.py::Critic`)
   - Maps features → 512 → 1
   - Outputs state value estimate V(s)
   - Provides baseline for advantage calculation

4. **A2CAgent** (`networks.py::A2CAgent`)
   - Wrapper combining all three components
   - Provides unified interface for training

## Algorithm

### Training Loop

1. **Collect Trajectory**
   - Run policy for N_STEPS (default: 5)
   - Store states, actions, rewards, values, log_probs

2. **Compute Advantages**
   - Use Generalized Advantage Estimation (GAE)
   - GAE balances bias-variance tradeoff
   - Formula: `A_t = δ_t + γλ·A_{t+1}` where `δ_t = r_t + γ·V(s_{t+1}) - V(s_t)`

3. **Calculate Losses**
   - **Policy Loss**: `-E[log π(a|s) × A(s,a)]` (maximize expected return)
   - **Value Loss**: `MSE(V(s), returns)` (minimize prediction error)
   - **Entropy Loss**: `-H(π)` (encourage exploration)
   - **Total Loss**: `policy_loss + 0.5×value_loss - 0.01×entropy`

4. **Update Network**
   - Single backward pass updates both actor and critic
   - Gradient clipping prevents instability

5. **Repeat** until mean reward ≥ 19.5

## Hyperparameters

### A2C-Specific
- `N_STEPS = 5` - Trajectory length per rollout
- `VALUE_LOSS_COEF = 0.5` - Weight for value loss
- `ENTROPY_COEF = 0.01` - Weight for entropy bonus
- `GAE_LAMBDA = 0.95` - GAE λ parameter (bias-variance)
- `MAX_GRAD_NORM = 0.5` - Gradient clipping threshold

### Reused from DQN
- `LEARNING_RATE = 0.0001` - Adam learning rate
- `GAMMA = 0.99` - Discount factor
- `FRAME_SKIP = 4` - Action repeat
- `INPUT_CHANNELS = 2` - Stacked frames
- `MEAN_REWARD_BOUND = 19.5` - Success threshold

## File Structure

```
pong_a2c_cnn/
├── README.md                    # This file
├── pong_a2c_cnn_v1.ipynb       # Main training notebook
├── networks.py                  # Actor-Critic architecture
├── a2c_utils.py                # A2C utilities (GAE, rollouts, losses)
└── logging_utils.py            # MLflow/WandB logging (adapted from DQN)
```

## Component Reuse from DQN

### 100% Reused
- Frame preprocessing pipeline (`preprocess_frame`, `FrameStack`)
- CNN feature extractor architecture (32→36→20 filters)
- Logging infrastructure (MLflow + WandB integration)
- Environment setup and frame skipping
- Hyperparameter management pattern

### Removed
- Replay buffer (A2C is on-policy)
- Target network (not needed for policy gradients)
- ε-greedy exploration (replaced by entropy)
- Double Q-learning logic

### New Components
- Actor and Critic network heads
- Trajectory buffer (sequential, not random sampling)
- GAE advantage estimation
- Policy gradient loss computation
- A2C-specific logging (policy/value/entropy losses)

## Model Size

Approximate parameter count:
- **Shared CNN**: ~29K parameters (2.1%)
- **Actor Head**: ~660K parameters (48%)
- **Critic Head**: ~660K parameters (48%)
- **Total**: ~1.35M parameters

Compared to Double DQN (~1.06M), A2C has ~28% more parameters due to separate actor and critic heads vs single Q-value head.

## Usage

### Running Training

```python
# In Jupyter notebook or Python script
from networks import A2CAgent
from a2c_utils import *
from logging_utils import *

# Load notebook: pong_a2c_cnn_v1.ipynb
# Run all cells to start training
```

### Monitoring

Training is logged to:
- **MLflow**: `/rl_study/mlruns/`
- **WandB**: Cloud dashboard + local `/rl_study/wandb/`
- **Artifacts**: `/rl_study/artifacts/run_{run_id}/`

Metrics tracked:
- Episode rewards and mean rewards
- Policy loss, value loss, entropy loss
- Advantage statistics
- Gradient norms
- Weight distributions

## Expected Performance

Based on the DQN baseline:
- **Convergence**: Should solve (mean reward ≥ 19.5) within similar episode count as DQN
- **Stability**: A2C typically has more stable training than DQN
- **Sample Efficiency**: May require fewer total samples due to on-policy learning
- **Speed**: Slightly slower per update (~28% more params) but fewer updates needed (no replay buffer overhead)

## Extending the Implementation

### Easy Extensions
1. **A3C**: Add asynchronous workers (multi-process)
2. **PPO**: Add clipped surrogate objective and multiple epochs
3. **Different N_STEPS**: Tune rollout length (5→20)
4. **Entropy Scheduling**: Decay entropy coefficient over time

### Advanced Extensions
1. **Recurrent Networks**: Add LSTM for partial observability
2. **Curiosity-Driven**: Add intrinsic motivation
3. **Multi-Objective**: Balance multiple reward signals
4. **Distributional**: Learn value distribution instead of mean

## References

- **A2C Paper**: [Asynchronous Methods for Deep RL (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783)
- **GAE Paper**: [High-Dimensional Continuous Control Using GAE (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
- **Base DQN**: `../pong_double_dqn_cnn/` directory

## License

Part of the RL Study project.
