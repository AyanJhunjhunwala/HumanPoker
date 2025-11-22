"""
Inference script for trained DQN Poker Agent
"""
from orchestrator import PokerTrainer, Config

if __name__ == "__main__":
    # Load config and model
    config = Config()
    trainer = PokerTrainer(config)
    
    # Load checkpoint
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "poker_dqn_checkpoint.pt"
    trainer.load(checkpoint_path)
    
    # Run inference on a few hands
    print("Running inference on 5 hands...")
    for i in range(5):
        state = trainer.env.reset()
        done = False
        step = 0
        
        print(f"\nHand {i+1}:")
        while not done and step < 20:  # Limit steps
            action_idx, (action_type, amount), probs = trainer.inference(state)
            print(f"  Step {step}: {action_type.upper()} (amount: {amount})")
            print(f"    Policy: Fold={probs[0]:.2f}, Call={probs[1]:.2f}, Raise={probs[2]:.2f}")
            
            state, reward, done, info = trainer.env.step(0, action_type, amount)
            if done:
                print(f"  Hand finished. Reward: {reward:.2f}, Pot: {info['pot']}")
            step += 1

