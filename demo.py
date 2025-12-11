#!/usr/bin/env python3
"""
Demo script for Poker DQN Agent - Orchestrates training and inference.

Usage:
    python demo.py train              # Train a new model
    python demo.py train --steps 10000 --save model.pt # USE THIS FOR SOMETHING QUICKER, adjust steps accordingly however
    python demo.py run                # Run inference with default checkpoint
    python demo.py run --checkpoint model.pt --hands 10
    python demo.py interactive        # Interactive play mode
    python demo.py evaluate           # Evaluate a trained model
"""

import argparse
import sys
import os
from orchestrator import PokerTrainer, Config


def train(args):
    """Train the poker DQN agent."""
    print("=" * 50)
    print("  POKER DQN TRAINING")
    print("=" * 50)
    
    config = Config(
        train_steps=args.steps,
        eval_freq=args.eval_freq,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        gamma=args.gamma,
        alpha_human=args.alpha_human,
    )
    
    print(f"\nConfiguration:")
    print(f"  Training steps: {config.train_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Epsilon: {config.epsilon_start} -> {config.epsilon_end}")
    print(f"  Human imitation weight (alpha): {config.alpha_human}")
    print()
    
    trainer = PokerTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load(args.resume)
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    save_path = args.save or "poker_dqn_checkpoint.pt"
    trainer.save(save_path)
    print(f"\n✓ Training complete! Model saved to {save_path}")
    
    return trainer


def run_inference(args):
    """Run inference on trained model."""
    print("=" * 50)
    print("  POKER DQN INFERENCE")
    print("=" * 50)
    
    config = Config()
    trainer = PokerTrainer(config)
    
    checkpoint_path = args.checkpoint or "model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first with: python demo.py train")
        sys.exit(1)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    trainer.load(checkpoint_path)
    
    num_hands = args.hands
    print(f"\nRunning inference on {num_hands} hands...\n")
    
    total_reward = 0
    wins = 0
    
    for i in range(num_hands):
        state = trainer.env.reset()
        done = False
        step = 0
        hand_reward = 0
        
        print(f"{'='*40}")
        print(f"HAND {i + 1}")
        print(f"{'='*40}")
        
        # Show hole cards
        hole_cards = trainer.env.player_cards[0]
        print(f"Hole Cards: {hole_cards[0][0]}{hole_cards[0][1]} {hole_cards[1][0]}{hole_cards[1][1]}")
        print()
        
        while not done and step < 20:
            action_idx, (action_type, amount), probs = trainer.inference(state)
            
            # Display action probabilities
            prob_str = f"[Fold: {probs[0]:.1%}, Call: {probs[1]:.1%}, Raise: {probs[2]:.1%}]"
            
            if action_type == 'fold':
                print(f"  Action: FOLD {prob_str}")
            elif action_type == 'call':
                print(f"  Action: CALL ${amount} {prob_str}")
            else:
                print(f"  Action: RAISE ${amount} {prob_str}")
            
            state, reward, done, info = trainer.env.step(0, action_type, amount)
            hand_reward += reward
            
            # Show community cards if available
            if trainer.env.community_cards and step == 0:
                comm = trainer.env.community_cards
                comm_str = ' '.join([f"{c[0]}{c[1]}" for c in comm])
                print(f"  Community: {comm_str}")
            
            step += 1
        
        total_reward += hand_reward
        if hand_reward > 0:
            wins += 1
        
        print(f"\n  Result: {'WIN' if hand_reward > 0 else 'LOSS'} (${hand_reward:+.0f})")
        print(f"  Pot: ${info['pot']}, Stack: ${info['stack']}")
        print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Hands played: {num_hands}")
    print(f"Wins: {wins} ({wins/num_hands:.1%})")
    print(f"Total profit: ${total_reward:+.0f}")
    print(f"Avg profit/hand: ${total_reward/num_hands:+.2f}")
    
    return trainer


def interactive_mode(args):
    """Interactive play mode where user can see AI decisions."""
    print("=" * 50)
    print("  POKER DQN - INTERACTIVE MODE")
    print("=" * 50)
    
    config = Config()
    trainer = PokerTrainer(config)
    
    checkpoint_path = args.checkpoint or "model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first with: python demo.py train")
        sys.exit(1)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    trainer.load(checkpoint_path)
    
    print("\nInteractive Mode - Press Enter to see each action, 'q' to quit\n")
    
    hand_num = 0
    total_profit = 0
    
    while True:
        hand_num += 1
        state = trainer.env.reset()
        done = False
        step = 0
        hand_reward = 0
        
        print(f"\n{'='*50}")
        print(f"HAND #{hand_num}")
        print(f"{'='*50}")
        
        # Show hole cards
        hole_cards = trainer.env.player_cards[0]
        print(f"Your Cards: {hole_cards[0][0]}{hole_cards[0][1]} {hole_cards[1][0]}{hole_cards[1][1]}")
        print(f"Stack: ${trainer.env.stacks[0]}, Pot: ${trainer.env.pot}")
        
        while not done and step < 20:
            # Wait for user input
            user_input = input("\n[Press Enter for AI action, 'q' to quit] > ").strip().lower()
            if user_input == 'q':
                print(f"\n{'='*50}")
                print(f"SESSION SUMMARY")
                print(f"{'='*50}")
                print(f"Hands played: {hand_num - 1}")
                print(f"Total profit: ${total_profit:+.0f}")
                return trainer
            
            action_idx, (action_type, amount), probs = trainer.inference(state)
            
            # Show community cards
            if trainer.env.community_cards:
                comm = trainer.env.community_cards
                comm_str = ' '.join([f"{c[0]}{c[1]}" for c in comm])
                print(f"Board: {comm_str}")
            
            print(f"\nAI Analysis:")
            print(f"  Fold probability:  {probs[0]:6.1%}")
            print(f"  Call probability:  {probs[1]:6.1%}")
            print(f"  Raise probability: {probs[2]:6.1%}")
            
            if action_type == 'fold':
                print(f"\n>>> AI FOLDS")
            elif action_type == 'call':
                print(f"\n>>> AI CALLS ${amount}")
            else:
                print(f"\n>>> AI RAISES to ${amount}")
            
            state, reward, done, info = trainer.env.step(0, action_type, amount)
            hand_reward += reward
            step += 1
            
            print(f"Pot: ${info['pot']}, Stack: ${info['stack']}")
        
        total_profit += hand_reward
        print(f"\n{'─'*30}")
        print(f"Hand Result: {'WIN' if hand_reward > 0 else 'LOSS'} (${hand_reward:+.0f})")
        print(f"Session Profit: ${total_profit:+.0f}")
        
        continue_input = input("\nPlay another hand? [Enter/q] > ").strip().lower()
        if continue_input == 'q':
            break
    
    print(f"\n{'='*50}")
    print(f"SESSION SUMMARY")
    print(f"{'='*50}")
    print(f"Hands played: {hand_num}")
    print(f"Total profit: ${total_profit:+.0f}")
    
    return trainer


def evaluate(args):
    """Evaluate a trained model."""
    print("=" * 50)
    print("  POKER DQN EVALUATION")
    print("=" * 50)
    
    config = Config()
    trainer = PokerTrainer(config)
    
    checkpoint_path = args.checkpoint or "model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first with: python demo.py train")
        sys.exit(1)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    trainer.load(checkpoint_path)
    
    print(f"\nRunning evaluation ({args.episodes} episodes)...\n")
    
    # Run multiple evaluation rounds
    all_results = []
    for i in range(args.episodes):
        if i > 0:
            print(f"\n--- Evaluation Round {i + 1}/{args.episodes} ---")
        results = trainer.evaluate()
        all_results.append(results)
    
    # Aggregate results
    if args.episodes > 1:
        print("\n" + "=" * 50)
        print("AGGREGATE RESULTS")
        print("=" * 50)
        
        avg_winrate = sum(r['winrate'] for r in all_results) / len(all_results)
        avg_reward = sum(r['avg_reward'] for r in all_results) / len(all_results)
        avg_bb100 = sum(r['bb_per_100'] for r in all_results) / len(all_results)
        
        print(f"Average Winrate: {avg_winrate:.2%}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average BB/100: {avg_bb100:.2f}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Poker DQN Agent - Training and Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py train                          # Train with defaults
  python demo.py train --steps 100000           # Train for 100k steps
  python demo.py train --steps 50000 --save my_model.pt
  python demo.py run                            # Run inference
  python demo.py run --hands 20                 # Run 20 hands
  python demo.py interactive                    # Interactive mode
  python demo.py evaluate --episodes 5          # Evaluate 5 times
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the poker agent")
    train_parser.add_argument("--steps", type=int, default=50000, 
                              help="Number of training steps (default: 50000)")
    train_parser.add_argument("--eval-freq", type=int, default=5000,
                              help="Evaluation frequency (default: 5000)")
    train_parser.add_argument("--batch-size", type=int, default=64,
                              help="Batch size (default: 64)")
    train_parser.add_argument("--lr", type=float, default=1e-4,
                              help="Learning rate (default: 1e-4)")
    train_parser.add_argument("--gamma", type=float, default=0.99,
                              help="Discount factor (default: 0.99)")
    train_parser.add_argument("--epsilon-start", type=float, default=1.0,
                              help="Starting epsilon (default: 1.0)")
    train_parser.add_argument("--epsilon-end", type=float, default=0.01,
                              help="Ending epsilon (default: 0.01)")
    train_parser.add_argument("--alpha-human", type=float, default=0.1,
                              help="Human imitation weight (default: 0.1)")
    train_parser.add_argument("--save", type=str, default=None,
                              help="Path to save model (default: poker_dqn_checkpoint.pt)")
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Path to checkpoint to resume from")
    
    # Run/Inference subcommand
    run_parser = subparsers.add_parser("run", help="Run inference with trained model")
    run_parser.add_argument("--checkpoint", type=str, default=None,
                            help="Path to model checkpoint (default: poker_dqn_checkpoint.pt)")
    run_parser.add_argument("--hands", type=int, default=5,
                            help="Number of hands to play (default: 5)")
    
    interactive_parser = subparsers.add_parser("interactive", 
                                                help="Interactive play mode")
    interactive_parser.add_argument("--checkpoint", type=str, default=None,
                                    help="Path to model checkpoint")
    
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--checkpoint", type=str, default=None,
                             help="Path to model checkpoint")
    eval_parser.add_argument("--episodes", type=int, default=1,
                             help="Number of evaluation episodes (default: 1)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python demo.py train    # Train a new model")
        print("   python demo.py run      # Run the trained model")
        sys.exit(0)
    
    if args.command == "train":
        train(args)
    elif args.command == "run":
        run_inference(args)
    elif args.command == "interactive":
        interactive_mode(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
