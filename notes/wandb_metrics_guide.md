# WandB Metrics Guide for A2C Defender Training

## Overview
This guide provides a comprehensive description of all metrics logged to Weights & Biases (wandb) during A2C training on Atari Defender. Understanding these metrics is crucial for diagnosing training issues, optimizing hyperparameters, and evaluating model performance.

---

## Table of Contents
1. [Model Architecture Metrics](#model-architecture-metrics)
2. [Training Performance Metrics](#training-performance-metrics)
3. [Loss Metrics](#loss-metrics)
4. [Gradient Metrics](#gradient-metrics)
5. [Weight Statistics](#weight-statistics)
6. [Final Results Metrics](#final-results-metrics)
7. [Metric Interpretation & Tradeoffs](#metric-interpretation--tradeoffs)
8. [Common Training Patterns](#common-training-patterns)

---

## Model Architecture Metrics
**Logged once at training start**

### `model_info/total_parameters`
- **Description**: Total number of trainable parameters in the entire A2C network
- **Typical Range**: 6-7 million for CNN-based A2C on Atari
- **Interpretation**:
  - Higher values → More model capacity but slower training
  - Lower values → Faster training but may underfit
- **What to Watch**: Ensure this matches your architecture design

### `model_info/trainable_parameters`
- **Description**: Number of parameters that will be updated during training
- **Typical Range**: Should equal total_parameters unless using frozen layers
- **Interpretation**: If less than total_parameters, some layers are frozen

### `model_info/conv_parameters`
- **Description**: Parameters in the shared convolutional feature extractor
- **Typical Range**: 50-70% of total parameters
- **Interpretation**:
  - High percentage → More capacity for visual feature extraction
  - Low percentage → More capacity in actor/critic heads

### `model_info/actor_parameters`
- **Description**: Parameters in the policy (actor) head
- **Typical Range**: 15-25% of total parameters
- **Interpretation**: Controls policy expressiveness

### `model_info/critic_parameters`
- **Description**: Parameters in the value (critic) head
- **Typical Range**: 15-25% of total parameters
- **Interpretation**: Controls value function approximation capacity

### `model_info/model_size_mb`
- **Description**: Model size in megabytes (assuming float32)
- **Typical Range**: 20-30 MB for Atari A2C
- **Interpretation**: Important for deployment and memory constraints

### `model_info/{component}_percentage`
- **Description**: Percentage of total parameters in each component (conv/actor/critic)
- **Typical Range**: conv ~50-70%, actor ~15-25%, critic ~15-25%
- **Interpretation**: Shows parameter distribution across architecture components

---

## Training Performance Metrics
**Logged every episode**

### `training/episode_reward`
- **Description**: Total undiscounted reward accumulated in the current episode
- **Typical Range**: For Defender, 0-20,000+ (highly variable)
- **Good Behavior**:
  - Gradual upward trend over time
  - Increasing variance as agent explores better strategies
  - Occasional spikes indicating discovery of new tactics
- **Bad Behavior**:
  - Completely flat (no learning)
  - Decreasing trend (catastrophic forgetting)
  - Extreme volatility without improvement (unstable learning)
- **Tradeoffs**:
  - High variance is normal in stochastic environments
  - Use mean_reward_100 for clearer trend signal

### `training/mean_reward_100`
- **Description**: Moving average of episode rewards over last 100 episodes
- **Typical Range**: Same as episode_reward but smoothed
- **Good Behavior**:
  - Steady upward trend
  - Plateaus followed by jumps (indicates learning phases)
  - Converges near or above reward threshold (15,000 for Defender)
- **Bad Behavior**:
  - No upward trend after 1000+ episodes
  - Sudden drops (possible catastrophic forgetting)
  - Oscillations without convergence
- **Tradeoffs**:
  - Smooths out noise but lags behind recent performance
  - Primary metric for evaluating training progress

### `training/episode_length`
- **Description**: Number of environment steps (after frame skip) in the episode
- **Typical Range**: 100-5,000+ steps for Defender
- **Good Behavior**:
  - Increases as agent learns survival strategies
  - Correlation with episode reward
- **Bad Behavior**:
  - Decreasing trend (agent dying faster)
  - No improvement over time
- **Tradeoffs**:
  - Longer episodes → more experience per episode
  - Very long episodes may indicate conservative play

### `training/episode`
- **Description**: Current episode number (training iteration)
- **Interpretation**: X-axis for most training plots

### `system/total_updates`
- **Description**: Total number of gradient update steps performed
- **Typical Range**: Can be much higher than episode count due to multiple updates per episode
- **Interpretation**:
  - Better measure of training progress than episode count
  - More updates per episode → more sample-efficient but slower wall-clock time
  - Use as x-axis for loss and gradient plots

---

## Loss Metrics
**Logged every episode**

### `losses/policy_loss`
- **Description**: Negative mean of (log_prob * advantage) - measures policy gradient
- **Typical Range**: -10 to 10 (highly dependent on advantage scaling)
- **Good Behavior**:
  - Starts high, decreases and stabilizes
  - Some fluctuation is normal due to stochastic policy
  - Negative values are expected (maximizing expected return)
- **Bad Behavior**:
  - Consistently increasing trend (policy degrading)
  - Exploding values (gradient instability)
  - Stuck at zero (no policy updates)
- **Tradeoffs**:
  - High values → Large policy changes (fast learning but unstable)
  - Low values → Small policy changes (stable but slow learning)
  - Related to advantage normalization and learning rate

### `losses/value_loss`
- **Description**: MSE between predicted state values and actual returns
- **Typical Range**: 0.1 to 1000+ (depends on reward scale)
- **Good Behavior**:
  - Decreases over time as value function improves
  - Stabilizes at low value (good value approximation)
  - May increase temporarily when discovering new high-reward states
- **Bad Behavior**:
  - Continuously increasing (value function diverging)
  - Not decreasing after many updates (poor learning)
  - Sudden spikes (instability)
- **Tradeoffs**:
  - High loss → Poor value estimates → Inaccurate advantages → Slow policy learning
  - Very low loss doesn't guarantee good policy (could be overfitting)
  - Scaled by VALUE_LOSS_COEF (0.5) in total loss

### `losses/entropy_loss`
- **Description**: Negative entropy of policy distribution
- **Typical Range**: Negative values, magnitude depends on action space size
- **Good Behavior**:
  - Starts high (large entropy = more exploration)
  - Gradually becomes more negative (decreasing entropy = more deterministic)
  - Stabilizes at moderate negative value (balanced exploration/exploitation)
- **Bad Behavior**:
  - Too negative too fast (premature convergence to deterministic policy)
  - Stays near zero (policy not becoming more decisive)
  - Extreme values (numerical instability)
- **Tradeoffs**:
  - High entropy → More exploration, slower convergence, more robust
  - Low entropy → More exploitation, faster convergence, risk of local optima
  - Controlled by ENTROPY_COEF (0.01) hyperparameter

### `losses/total_loss`
- **Description**: policy_loss + VALUE_LOSS_COEF * value_loss + ENTROPY_COEF * entropy_loss
- **Typical Range**: Highly variable, depends on components
- **Good Behavior**:
  - Decreasing trend overall
  - Stabilizes after convergence
- **Bad Behavior**:
  - Increasing trend
  - Extreme fluctuations
- **Tradeoffs**:
  - Dominated by largest component (usually value_loss or policy_loss)
  - Balance controlled by loss coefficients

---

## Gradient Metrics
**Logged every GRADIENT_LOG_FREQ updates (default: 500)**

### `grad_norms/{layer_name}`
- **Description**: L2 norm of gradients for each network layer
- **Typical Layers**: conv.0.weight, conv.2.weight, actor_fc.weight, policy_head.weight, critic_fc.weight, value_head.weight
- **Typical Range**: 0.001 to 10.0 (depends on layer and gradient clipping)
- **Good Behavior**:
  - Gradients present in all layers (no vanishing gradients)
  - Magnitudes within 0.01 to 5.0 range
  - Relatively stable over time (after initial training phase)
  - Similar magnitudes across layers (balanced learning)
- **Bad Behavior**:
  - **Vanishing gradients**: Values < 1e-6 in early layers
    - Symptom: Only final layers learning
    - Solution: Check learning rate, use skip connections, adjust initialization
  - **Exploding gradients**: Values > 10.0 despite clipping
    - Symptom: Training instability, NaN losses
    - Solution: Lower learning rate, increase gradient clipping, check architecture
  - **Dead gradients**: Consistent zeros in some layers
    - Symptom: Layer not learning
    - Solution: Check activation functions (dying ReLU), initialization
  - **Wildly fluctuating**: Large variance in gradient norms
    - Symptom: Unstable training
    - Solution: Reduce learning rate, increase batch size (N_STEPS)
- **Tradeoffs**:
  - Higher gradients → Faster learning but less stable
  - Lower gradients → More stable but slower learning
  - Gradient clipping (MAX_GRAD_NORM=0.5) prevents exploding gradients

---

## Weight Statistics
**Logged every LOG_FREQ episodes (default: 100)**

### `weight_stats/{layer_name}_mean`
- **Description**: Mean of weight values in the layer
- **Typical Range**: Close to 0 for well-initialized networks
- **Good Behavior**:
  - Stays near zero
  - Slight drift is acceptable
- **Bad Behavior**:
  - Drifting far from zero (layer bias developing)
  - Sudden large changes

### `weight_stats/{layer_name}_std`
- **Description**: Standard deviation of weight values
- **Typical Range**: 0.01 to 0.5 depending on layer size and initialization
- **Good Behavior**:
  - Relatively stable
  - Appropriate for layer size (smaller for larger layers)
- **Bad Behavior**:
  - Decreasing to near zero (weights collapsing)
  - Exploding (unbounded weight growth)

### `weight_stats/{layer_name}_min` / `_max`
- **Description**: Minimum and maximum weight values in layer
- **Typical Range**: -3.0 to 3.0 for well-behaved networks
- **Good Behavior**:
  - Bounded within reasonable range
  - Symmetric distribution (min ≈ -max)
- **Bad Behavior**:
  - Extreme outliers (|value| > 10)
  - Asymmetric (suggests systematic bias)

### `weight_stats/{layer_name}_norm`
- **Description**: L2 norm of all weights in the layer
- **Typical Range**: 1.0 to 100.0 depending on layer size
- **Good Behavior**:
  - Stable or slowly changing
  - Proportional to layer size
- **Bad Behavior**:
  - Rapid growth (weight explosion)
  - Approaching zero (weight decay too strong)

### `weight_stats/{layer_name}_sparsity`
- **Description**: Fraction of weights with |value| < 1e-6
- **Typical Range**: 0.0 to 0.1 for active layers
- **Good Behavior**:
  - Low sparsity (most weights being used)
  - Slight increase over training (natural pruning)
- **Bad Behavior**:
  - High sparsity (> 0.5) early in training
  - Rapidly increasing (network dying)

---

## Final Results Metrics
**Logged once at end of training**

### `final_episode`
- **Description**: Total number of episodes completed
- **Interpretation**: Training duration measure

### `final_mean_reward`
- **Description**: Final mean reward over last 100 episodes
- **Interpretation**: Primary success metric - should be ≥ MEAN_REWARD_BOUND (15,000)

### `total_episodes`
- **Description**: Same as final_episode
- **Interpretation**: Redundant with final_episode

### `solved_at_episode`
- **Description**: Episode number when mean_reward_100 first reached threshold
- **Interpretation**: Sample efficiency metric - lower is better

---

## Metric Interpretation & Tradeoffs

### Policy Learning vs. Value Learning Balance

**Scenario 1: Policy loss high, value loss decreasing**
- **Meaning**: Value function learning well but policy struggling
- **Action**: Check advantage normalization, increase policy learning rate coefficient

**Scenario 2: Policy loss decreasing, value loss high**
- **Meaning**: Policy improving despite poor value estimates (lucky exploration)
- **Risk**: Unstable - poor value estimates lead to poor policy updates
- **Action**: Focus on improving value function - increase VALUE_LOSS_COEF, check network capacity

**Scenario 3: Both losses high**
- **Meaning**: Network struggling to learn both tasks
- **Action**: Check learning rate, network capacity, ensure gradients flowing

**Scenario 4: Both losses low but no reward improvement**
- **Meaning**: Network converged to local optimum or overfitting
- **Action**: Increase exploration (ENTROPY_COEF), reset training, adjust architecture

### Exploration vs. Exploitation Trade-off

**High Entropy (less negative entropy_loss)**
- **Pros**: More exploration, robust to local optima, better generalization
- **Cons**: Slower convergence, lower peak performance, noisy behavior
- **When to use**: Early training, complex environments, when stuck in local optimum

**Low Entropy (more negative entropy_loss)**
- **Pros**: Faster convergence, higher peak performance, consistent behavior
- **Cons**: Risk of premature convergence, brittle to changes, local optima
- **When to use**: Fine-tuning, near-optimal policy, simple environments

### Sample Efficiency vs. Stability

**More updates per episode (higher total_updates/episode ratio)**
- **Pros**: Better sample efficiency, faster learning per episode
- **Cons**: Risk of overfitting to recent experience, slower wall-clock time
- **Controlled by**: N_STEPS (rollout length)

**Fewer updates per episode**
- **Pros**: More stable, less overfitting risk, faster wall-clock time
- **Cons**: Slower learning, less sample efficient
- **Controlled by**: N_STEPS (rollout length)

---

## Common Training Patterns

### Pattern 1: Healthy Training
- **episode_reward**: Upward trend with high variance
- **mean_reward_100**: Smooth upward curve
- **policy_loss**: Decreases then stabilizes with fluctuations
- **value_loss**: Decreases and stabilizes at low value
- **entropy_loss**: Gradually becomes more negative
- **grad_norms**: Stable, present in all layers
- **Action**: Continue training, consider minor hyperparameter tuning

### Pattern 2: No Learning (Flat Performance)
- **episode_reward**: No improvement, random fluctuations
- **mean_reward_100**: Flat line
- **policy_loss**: Not decreasing
- **value_loss**: High and not decreasing
- **Possible Causes**:
  - Learning rate too low
  - Network capacity insufficient
  - Vanishing gradients
  - Poor initialization
- **Action**: Increase learning rate, check gradients, verify architecture

### Pattern 3: Unstable Training
- **episode_reward**: Wild oscillations
- **mean_reward_100**: Oscillates without clear trend
- **policy_loss**: Erratic, large spikes
- **value_loss**: Spikes and instability
- **grad_norms**: Hitting gradient clip frequently
- **Possible Causes**:
  - Learning rate too high
  - Poor advantage normalization
  - Insufficient gradient clipping
- **Action**: Reduce learning rate, increase MAX_GRAD_NORM, stabilize advantages

### Pattern 4: Catastrophic Forgetting
- **mean_reward_100**: Increases then suddenly drops
- **policy_loss**: Sudden increase after being stable
- **value_loss**: Sudden increase after being stable
- **Possible Causes**:
  - Plasticity loss
  - Rare devastating experience
  - Learning rate too high
- **Action**: Reduce learning rate, use experience replay, adjust value clipping

### Pattern 5: Premature Convergence
- **mean_reward_100**: Plateaus well below target
- **entropy_loss**: Becomes very negative early
- **policy_loss**: Near zero
- **value_loss**: Stable but suboptimal
- **Possible Causes**:
  - Entropy coefficient too low
  - Local optimum
  - Insufficient exploration
- **Action**: Increase ENTROPY_COEF, restart training, add exploration noise

### Pattern 6: Slow but Steady Learning
- **mean_reward_100**: Very gradual upward trend
- **All losses**: Slowly decreasing
- **grad_norms**: Very small but present
- **Possible Causes**:
  - Learning rate too low (conservative)
  - Very stable but inefficient training
- **Action**: Consider increasing learning rate for faster training

---

## Hyperparameter Impact on Metrics

### LEARNING_RATE (0.0001)
- **Affects**: All losses, gradient norms, weight statistics
- **Higher** → Faster changes in all metrics, more instability
- **Lower** → Slower changes, more stability

### GAMMA (0.99)
- **Affects**: Value loss, policy loss (through returns)
- **Higher** → Longer-term credit assignment, higher value estimates
- **Lower** → Shorter-term focus, more myopic policy

### N_STEPS (5)
- **Affects**: Updates per episode, bias-variance trade-off
- **Higher** → Less bias, more variance, fewer updates per episode
- **Lower** → More bias, less variance, more updates per episode

### VALUE_LOSS_COEF (0.5)
- **Affects**: Total loss composition, learning balance
- **Higher** → More emphasis on value learning
- **Lower** → More emphasis on policy learning

### ENTROPY_COEF (0.01)
- **Affects**: Entropy loss magnitude, exploration level
- **Higher** → More exploration, slower convergence
- **Lower** → Less exploration, faster convergence

### MAX_GRAD_NORM (0.5)
- **Affects**: Gradient norms (caps them), training stability
- **Higher** → Allows larger gradient steps, less clipping
- **Lower** → More aggressive clipping, more stability

---

## Monitoring Checklist

### Every 100 Episodes
- [ ] Check mean_reward_100 is improving
- [ ] Verify weight statistics are stable
- [ ] Check for dead gradients (periodic)

### Every 500 Episodes
- [ ] Review gradient norms across all layers
- [ ] Check loss trends
- [ ] Verify no exploding/vanishing gradients

### Every 1000 Episodes
- [ ] Analyze training_plot artifact
- [ ] Review all loss components
- [ ] Check entropy level appropriate for training phase
- [ ] Verify episode_length improving with reward

### End of Training
- [ ] Final mean reward ≥ target (15,000)
- [ ] Training converged (stable losses)
- [ ] No signs of instability in final episodes
- [ ] Gradient norms healthy across all layers

---

## Quick Diagnostic Guide

| Symptom | Likely Cause | Check Metrics | Solution |
|---------|--------------|---------------|----------|
| No learning | LR too low, arch issues | grad_norms, value_loss | Increase LR, check gradients |
| Unstable | LR too high | policy_loss, value_loss spikes | Decrease LR, clip gradients |
| Plateaus early | Local optimum | entropy_loss | Increase ENTROPY_COEF |
| Slow learning | Conservative hyperparams | All losses decreasing slowly | Increase LR moderately |
| Reward drops | Catastrophic forgetting | Sudden loss spikes | Reduce LR, check recent experiences |
| High variance | Poor value estimates | value_loss high | Increase VALUE_LOSS_COEF |

---

## Additional Resources

- **wandb Documentation**: https://docs.wandb.ai/
- **A2C Algorithm**: https://arxiv.org/abs/1602.01783
- **Gradient Diagnostics**: Check `grad_norms/*` for vanishing/exploding gradients
- **Training Plots**: Review `training_plot` artifact for visual trends

---

**Last Updated**: Based on defender_a2c_cnn_v1 implementation
**Hyperparameters Referenced**: LR=0.0001, GAMMA=0.99, N_STEPS=5, VALUE_LOSS_COEF=0.5, ENTROPY_COEF=0.01, MAX_GRAD_NORM=0.5
