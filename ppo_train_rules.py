"""Opt-in PPO entry for the versioned candidate rules; legacy entry is unchanged."""
import ppo_train
from training.rules_candidate_runtime import install_training

if __name__ == '__main__':
    install_training(ppo_train)
    ppo_train.main()
