### 25/10/2025

> why the mean reward for the training of defender is oscillating? Is that due to low capacity of the cnn?

Based on analyzing your training code, the oscillating mean reward for Defender is not due to low CNN capacity. 

Here are the actual reasons:

Main Causes of Oscillation:

* Game Complexity: Defender is significantly more complex than Pong:
    * Larger action space (18 actions vs Pong's 6)
    * More complex game mechanics (scrolling, multiple enemies, rescue missions)
    * Higher visual complexity and state space
* Insufficient Training Stability:
    * Learning rate (0.0001) may be too high for Defender's complexity
    * Target update frequency (500 episodes) is quite high - could cause instability
    * Batch size (32) is relatively small for complex patterns
* Epsilon Decay Issues:
    * Adaptive epsilon decay at convergence_threshold=10000 may be premature
    * Normal decay rate (0.995) might reduce exploration too quickly
* Network Architecture is Actually Fine:
    * Your CNN has ~1.4M parameters (32→64→64 filters + 512 FC units)
    * This is standard for DQN on Atari and sufficient for Defender

The issue is training dynamics, not capacity

Recommended Fixes:

* Lower learning rate: Try 0.00025 or 0.00005
* More frequent target updates: Try TARGET_UPDATE = 1000 (in steps, not episodes)
* Larger batch size: Try BATCH_SIZE = 64 or 128
* Higher convergence threshold: Increase to 50000+ for Defender
* Consider using Double DQN or Dueling DQN for better stability
* The CNN capacity is appropriate - the oscillation stems from hyperparameter tuning needs for this specific game's complexity.