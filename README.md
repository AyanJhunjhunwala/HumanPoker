# Human-Like Poker DQN Agent

A Deep Q-Network (DQN) poker agent that learns to play Texas Hold'em while imitating human-like playing styles through behavior cloning.

## Features

- **Dueling Double DQN** with prioritized experience replay
- **Behavior Cloning** to learn human-like play patterns
- **Human imitation reward** - rewards actions that match human policy
- **Opponent modeling** with simulated opponent strategies
- **Full hand evaluation** for accurate showdown results

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Model

Train a new poker agent:

```bash
# Default training (50,000 steps)
python demo.py train

# Quick training (lower quality, but faster)
python demo.py train --steps 10000 --save model.pt

# Full training with custom parameters
python demo.py train --steps 100000 --lr 0.0001 --batch-size 128 --save my_model.pt
```

**Training Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 50000 | Number of training steps |
| `--eval-freq` | 5000 | Evaluation frequency |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--epsilon-start` | 1.0 | Starting exploration rate |
| `--epsilon-end` | 0.01 | Final exploration rate |
| `--alpha-human` | 0.1 | Human imitation weight |
| `--save` | poker_dqn_checkpoint.pt | Save path |
| `--resume` | None | Resume from checkpoint |

### Running Inference

Run the trained model on sample hands:

```bash
# Run 5 hands with default checkpoint
python demo.py run

# Run with specific checkpoint and number of hands
python demo.py run --checkpoint model.pt --hands 10
```

### Interactive Mode

Step through hands interactively to see AI decision-making:

```bash
python demo.py interactive

# With specific checkpoint
python demo.py interactive --checkpoint model.pt
```

In interactive mode:
- Press **Enter** to see the next AI action
- Press **q** to quit
- View action probabilities and pot odds

### Evaluating a Model

Run comprehensive evaluation with statistics:

```bash
# Single evaluation
python demo.py evaluate

# Multiple evaluation episodes
python demo.py evaluate --episodes 5 --checkpoint model.pt
```

Evaluation shows:
- **Winrate** - Percentage of hands won
- **BB/100** - Big blinds won per 100 hands
- **KL Divergence** - How close to human policy
- **Poker Stats** - VPIP, PFR, Aggression Factor

## Project Structure

```
HumanPoker/
├── demo.py          # Main entry point - orchestrates training and inference
├── orchestrator.py  # PokerTrainer, Environment, and training logic
├── model.py         # Neural network architectures (DQN, Behavior Cloning)
├── gui.py           # GUI interface (optional)
├── train.py         # Standalone training script
├── inference.py     # Standalone inference script
└── requirements.txt # Python dependencies
```

## Architecture

### DQN Model
- **Dueling architecture** - Separate value and advantage streams
- **Double DQN** - Reduces overestimation bias
- **Prioritized Experience Replay** - Focuses on important transitions

### Behavior Cloning
- Pre-trains on synthetic human-like data
- Learns conservative play with strong hands, folding weak hands
- Used to compute human imitation bonus during RL training

### Reward Function
```
R(s, a) = chips_won + α * log(π_human(a|s))
```
- Balances winning chips with playing like a human
- `α` controls the human imitation weight

## Example Output

```
========================================
HAND 1
========================================
Hole Cards: A♠ K♦

  Action: RAISE $30 [Fold: 0.1%, Call: 15.2%, Raise: 84.7%]
  Community: J♥ 10♠ 2♦

  Result: WIN ($+45)
  Pot: $0, Stack: $1045
```

## TODO

- [ ] Parse real hand histories → unified state format
- [ ] Extract features: cards, pot, position, stacks
- [ ] Compute opponent stats (VPIP, PFR, AF, WWSF)
- [ ] Add tilt / streak features
- [ ] Optional LSTM for DRQN (sequential decisions)
- [ ] Trait head (predict player archetype)
- [ ] Compare to human stats on held-out data
- [ ] Winrate vs baseline bots & clones
