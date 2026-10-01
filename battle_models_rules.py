"""Use the same candidate rules and multi-winner metrics for every policy."""
import battle_models
from training.rules_candidate_runtime import install_evaluation

if __name__ == '__main__':
    install_evaluation(battle_models)
    battle_models.main()
