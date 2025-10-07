# GridWorld Reinforcement Learning: Q-Learning vs Deep Q-Network (DQN)

A comprehensive reinforcement learning implementation comparing traditional Q-Learning with Deep Q-Networks (DQN) for navigating a multi-goal grid world environment with obstacles.

## Overview

This project demonstrates two approaches to reinforcement learning in a 2D grid world with multiple goals and obstacles:

1. **Q-Learning**: Traditional tabular method using a Q-table
2. **Deep Q-Network (DQN)**: Neural network-based Q-function approximation

Both agents learn to navigate from a fixed starting position to randomly assigned target goals while avoiding obstacles.

## Features

### Common Features
- **Multi-Goal Environment**: Multiple goal positions with one designated as target per episode
- **Dynamic Obstacles**: Configurable number of obstacle positions
- **Visualization**: Step-by-step visualization of agent's path and training metrics
- **Flexible Configuration**: Adjustable grid size, goals, and obstacles

### Q-Learning Specific
- **Q-Table**: Dictionary-based state-action value storage
- **Epsilon-Greedy**: Simple exploration strategy
- **Direct Lookup**: Fast action selection

### DQN Specific
- **Neural Network**: Deep learning-based Q-function approximation
- **Experience Replay**: Stabilized learning through random batch sampling
- **Target Network**: Separate target network for stable Q-value targets
- **GPU Support**: Automatic GPU acceleration when available
- **Generalization**: Can handle unseen states

## Environment Details

### GridWorld Environment
- **Grid Size**: Configurable (default: 10x10)
- **Agent**: Green circle, starts at position (0, 0)
- **Goals**: Numbered circles (target goal in gold, others in gray)
- **Obstacles**: Black squares that block movement
- **Actions**: 4 discrete actions (UP, RIGHT, DOWN, LEFT)
- **State Space**: Agent's (x, y) position
- **Rewards**:
  - +100 for reaching target goal
  - -1 for each step (encourages efficiency)

## Installation

### Requirements
```bash
# For Q-Learning
pip install gymnasium numpy matplotlib

# For DQN (additional requirement)
pip install torch
```

### Dependencies
- `gymnasium`: Reinforcement learning environment framework
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `torch`: PyTorch for neural networks (DQN only)

## Usage

### Q-Learning Implementation

```python
# Run Q-Learning training
python gridworld_qlearning.py
```

#### Custom Q-Learning Configuration
```python
env, agent, rewards, steps, goals = train_and_visualize(
    grid_size=10,          # Size of grid (10x10)
    num_goals=4,           # Number of goal positions
    num_obstacles=3,       # Number of obstacles
    episodes=500           # Training episodes
)
```

#### Q-Learning Hyperparameters
```python
agent = QLearningAgent(
    action_space_size=4,         # 4 possible actions
    learning_rate=0.1,           # Q-table update rate
    discount_factor=0.95,        # Gamma (future reward discount)
    epsilon=1.0,                 # Initial exploration rate
    epsilon_decay=0.995,         # Exploration decay per episode
    epsilon_min=0.01             # Minimum exploration rate
)
```

### DQN Implementation

```python
# Run DQN training
python gridworld_dqn.py
```

#### Custom DQN Configuration
```python
env, agent, rewards, steps, goals = train_and_visualize(
    grid_size=10,          # Size of grid (10x10)
    num_goals=4,           # Number of goal positions
    num_obstacles=3,       # Number of obstacles
    episodes=500           # Training episodes
)
```

#### DQN Hyperparameters
```python
agent = DQNAgent(
    state_size=2,                    # (x, y) coordinates
    action_size=4,                   # 4 possible actions
    learning_rate=0.001,             # Network learning rate
    discount_factor=0.95,            # Gamma (future reward discount)
    epsilon=1.0,                     # Initial exploration rate
    epsilon_decay=0.995,             # Exploration decay per episode
    epsilon_min=0.01,                # Minimum exploration rate
    buffer_size=10000,               # Replay buffer capacity
    batch_size=64,                   # Training batch size
    target_update_freq=10            # Target network update frequency
)
```

## Algorithms

### Q-Learning Algorithm

**Q-Table Update Rule**:
```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
```

**Training Process**:
1. Initialize Q-table as empty dictionary (defaultdict)
2. For each episode:
   - Select action using ε-greedy policy
   - Execute action and observe reward and next state
   - Update Q-table using Q-learning rule
   - Decay exploration rate ε
3. Repeat until convergence

**Advantages**:
- Simple to implement and understand
- Guaranteed convergence (with proper conditions)
- No hyperparameter tuning for network architecture
- Fast lookup and updates

**Limitations**:
- Must visit every state-action pair
- Memory grows with state space
- Cannot generalize to unseen states
- Inefficient for large/continuous state spaces

### Deep Q-Network (DQN) Algorithm

**Neural Network Architecture**:
```
Input Layer (2 neurons) → Hidden Layer 1 (128 neurons, ReLU)
                       → Hidden Layer 2 (128 neurons, ReLU)
                       → Output Layer (4 neurons, one per action)
```

**Training Process**:
1. **Experience Collection**:
   - Agent selects action using ε-greedy policy
   - Execute action and observe reward and next state
   - Store experience (s, a, r, s', done) in replay buffer

2. **Network Training**:
   - Sample random batch from replay buffer
   - Compute current Q-values: Q(s, a)
   - Compute target Q-values: r + γ × max Q_target(s', a')
   - Minimize MSE loss between current and target Q-values
   - Update Q-network using backpropagation

3. **Target Network Update**:
   - Every N steps, copy weights from Q-network to target network
   - Provides stable targets during training

4. **Exploration Decay**:
   - Gradually reduce ε to shift from exploration to exploitation

**Advantages**:
- Generalizes to unseen states
- Efficient for large state spaces
- Can handle continuous state spaces
- Scalable with network capacity

**Limitations**:
- Requires more hyperparameter tuning
- Longer training time
- Needs experience replay and target networks for stability
- More complex to implement

## Comparison: Q-Learning vs DQN

| Feature | Q-Learning | DQN |
|---------|-----------|-----|
| **Representation** | Q-table (dictionary) | Neural network |
| **State Space** | Must visit every state | Generalizes to unseen states |
| **Memory** | O(states × actions) | O(network parameters) |
| **Training Speed** | Fast for small spaces | Slower, needs batches |
| **Scalability** | Poor for large spaces | Excellent for large spaces |
| **Continuous States** | Not supported | Naturally handles |
| **Implementation** | Simple | Complex |
| **Stability** | Inherently stable | Requires replay + target net |
| **Convergence** | Guaranteed (theoretically) | Not guaranteed |
| **Best For** | Small, discrete spaces | Large, complex spaces |

## Output Files

### Q-Learning Output
- `gridworld_all_steps_part1.png`: Step-by-step agent trajectory
- `gridworld_qlearning_training.png`: Training metrics

### DQN Output
- `gridworld_dqn_steps_part1.png`: Step-by-step agent trajectory
- `gridworld_dqn_training.png`: Training metrics

### Training Metrics (Both)
- Episode rewards with 50-episode moving average
- Steps to goal with 50-episode moving average
- Exploration rate (epsilon) over episodes

## Hyperparameter Tuning Guide

### Q-Learning Hyperparameters

| Parameter | Range | Effect | Recommended |
|-----------|-------|--------|-------------|
| Learning Rate (α) | 0.01 - 0.5 | Update magnitude | 0.1 |
| Discount Factor (γ) | 0.9 - 0.99 | Future reward weight | 0.95 |
| Epsilon | 0.5 - 1.0 | Initial exploration | 1.0 |
| Epsilon Decay | 0.99 - 0.999 | Exploration reduction | 0.995 |
| Epsilon Min | 0.01 - 0.1 | Minimum exploration | 0.01 |

### DQN Hyperparameters

| Parameter | Range | Effect | Recommended |
|-----------|-------|--------|-------------|
| Learning Rate | 0.0001 - 0.01 | Network update rate | 0.001 |
| Batch Size | 32 - 128 | Training stability | 64 |
| Buffer Size | 1000 - 100000 | Experience diversity | 10000 |
| Target Update Freq | 5 - 50 | Target stability | 10 |
| Hidden Layer Size | 64 - 256 | Network capacity | 128 |
| Discount Factor (γ) | 0.9 - 0.99 | Future reward weight | 0.95 |
| Epsilon Decay | 0.99 - 0.999 | Exploration reduction | 0.995 |

## Performance Tips

### Q-Learning
1. **Increase Episodes**: 1000+ for complex grids
2. **Adjust Learning Rate**: Higher for faster learning, lower for stability
3. **Exploration**: Ensure sufficient exploration before decay

### DQN
1. **GPU Acceleration**: Automatically uses CUDA if available
2. **Batch Size**: Increase for faster training (if memory allows)
3. **Network Architecture**: Add layers/neurons for complex environments
4. **Buffer Size**: Increase for better experience diversity
5. **Episodes**: Train longer (1000+) for complex grids

## Training Example Output

### Q-Learning
```
Training Q-Learning Agent with Changing Target Goals...
Grid Size: 10x10
Number of Goals: 4
Number of Obstacles: 3
Episodes: 500

Episode 100/500 - Avg Reward: 32.45, Avg Steps: 67.55, Most Common Goal: 2, Epsilon: 0.605
Episode 200/500 - Avg Reward: 58.23, Avg Steps: 41.77, Most Common Goal: 1, Epsilon: 0.366
Episode 300/500 - Avg Reward: 75.89, Avg Steps: 24.11, Most Common Goal: 3, Epsilon: 0.221
Episode 400/500 - Avg Reward: 85.34, Avg Steps: 14.66, Most Common Goal: 0, Epsilon: 0.134
Episode 500/500 - Avg Reward: 90.12, Avg Steps: 9.88, Most Common Goal: 2, Epsilon: 0.081

✓ Training Complete!
```

### DQN
```
Training DQN Agent with Changing Target Goals...
Grid Size: 10x10
Number of Goals: 4
Number of Obstacles: 3
Episodes: 500
Using device: cuda

Episode 100/500 - Avg Reward: -45.23, Avg Steps: 146.23, Epsilon: 0.605, Buffer Size: 10000
Episode 200/500 - Avg Reward: 15.67, Avg Steps: 84.33, Epsilon: 0.366, Buffer Size: 10000
Episode 300/500 - Avg Reward: 42.11, Avg Steps: 57.89, Epsilon: 0.221, Buffer Size: 10000
Episode 400/500 - Avg Reward: 68.45, Avg Steps: 31.55, Epsilon: 0.134, Buffer Size: 10000
Episode 500/500 - Avg Reward: 81.22, Avg Steps: 18.78, Epsilon: 0.081, Buffer Size: 10000

✓ Training Complete!
```

## When to Use Each Method

### Use Q-Learning When:
- State space is small and discrete (< 10,000 states)
- Quick prototyping needed
- Simplicity and interpretability are important
- Computational resources are limited
- You need guaranteed convergence properties

### Use DQN When:
- State space is large or continuous
- Need generalization to unseen states
- Complex environment dynamics
- GPU resources available
- Willing to invest time in hyperparameter tuning

## Troubleshooting

### Q-Learning Issues

**Agent Not Learning**
- Increase training episodes
- Adjust learning rate (try 0.1 - 0.3)
- Check epsilon decay (may be too fast)
- Verify reward structure

**Q-Table Too Large**
- Reduce state space dimensionality
- Consider state abstraction
- Switch to DQN for large spaces

### DQN Issues

**Training Unstable**
- Decrease learning rate
- Increase target update frequency
- Increase batch size
- Check for reward scaling issues

**Out of Memory**
- Decrease batch size
- Reduce buffer size
- Reduce network hidden layer sizes

**Not Converging**
- Increase training episodes
- Adjust learning rate
- Check network architecture
- Verify experience replay is working

## Code Structure

### Q-Learning Files
```
gridworld_qlearning.py
├── MultiGoalGridWorld      # Environment class
├── QLearningAgent          # Q-Learning agent
└── train_and_visualize()   # Training loop
```

### DQN Files
```
gridworld_dqn.py
├── MultiGoalGridWorld      # Environment class (same)
├── QNetwork                # Neural network architecture
├── ReplayBuffer            # Experience replay buffer
├── DQNAgent                # DQN agent
└── train_and_visualize()   # Training loop
```

## Future Enhancements

### Q-Learning
- [ ] State aggregation for larger grids
- [ ] Eligibility traces (SARSA(λ))
- [ ] Function approximation with linear models

### DQN
- [ ] Double DQN to reduce overestimation
- [ ] Dueling DQN for better value estimation
- [ ] Prioritized Experience Replay
- [ ] Multi-step returns (n-step DQN)
- [ ] Noisy networks for exploration
- [ ] Rainbow DQN (combining improvements)

### Both
- [ ] Curriculum learning with increasing difficulty
- [ ] Multi-agent scenarios
- [ ] Dynamic obstacle movement
- [ ] Partial observability (POMDP)

## References

### Q-Learning
- [Watkins, C.J.C.H. (1989). Learning from Delayed Rewards. PhD thesis](https://www.cs.rhul.ac.uk/~chrisw/thesis.html)
- [Sutton & Barto (2018). Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

### DQN
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (van Hasselt et al., 2015)](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures (Wang et al., 2016)](https://arxiv.org/abs/1511.06581)
- [Rainbow: Combining Improvements in DRL (Hessel et al., 2017)](https://arxiv.org/abs/1710.02298)

## License

MIT License

## Author

Mohamed RedA Nkira

## Acknowledgments

Built with PyTorch, Gymnasium, NumPy, and Matplotlib

