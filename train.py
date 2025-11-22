from orchestrator import PokerTrainer, Config

if __name__ == "__main__":
    config = Config(
        train_steps=50000,
        eval_freq=5000,
        batch_size=64,
        lr=1e-4
    )
    
    trainer = PokerTrainer(config)
    
    trainer.train()
    
    trainer.save("poker_dqn_checkpoint.pt")
    print("\nTraining complete! Model saved to poker_dqn_checkpoint.pt")

