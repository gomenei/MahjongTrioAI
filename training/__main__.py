"""Unified entry: python -m training {bc,match,ppo} [options]."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("method", choices=("bc", "match", "ppo"))
    parser.add_argument("--rules", choices=("legacy", "candidate"), default="legacy",
                        help="PPO environment rules; candidate requires --game-mode south")
    parser.add_argument("--consistent-forward", action="store_true",
                        help="PPO: use matching rollout and gradient forward paths")
    if not sys.argv[1:] or sys.argv[1:] in (["--help"], ["-h"]):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    if args.method != "ppo" and (args.rules != "legacy" or args.consistent_forward):
        parser.error("--rules and --consistent-forward apply to PPO only")
    if args.method == "bc":
        from training import supervised as engine
    elif args.method == "match":
        from training import match_supervised as engine
    else:
        from training import ppo as engine
        if args.rules == "candidate":
            from training.rules_candidate_runtime import install_training
            install_training(engine)
        if args.consistent_forward:
            from training.policy_numerics import install_training
            install_training(engine)
    sys.argv = [f"training {args.method}", *remaining]
    engine.main()


if __name__ == "__main__":
    main()
