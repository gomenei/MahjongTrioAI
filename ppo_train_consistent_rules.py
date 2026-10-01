"""Candidate rules with matching training rollout/gradient forward paths."""
import ppo_train
from training.rules_candidate_runtime import install_training as install_rules
from training.policy_numerics import install_training as install_numerics

if __name__ == '__main__':
    install_rules(ppo_train)
    install_numerics(ppo_train)
    ppo_train.main()
