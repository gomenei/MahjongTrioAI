"""Unified entry: python -m evaluation {models,copilot} [options]."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("method", choices=("models", "copilot"))
    parser.add_argument("--rules", choices=("legacy", "candidate"), default="legacy",
                        help="Local model battle rules; candidate requires --game-mode south")
    if not sys.argv[1:] or sys.argv[1:] in (["--help"], ["-h"]):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    if args.method == "models":
        from evaluation import battle as engine
        if args.rules == "candidate":
            from training.rules_candidate_runtime import install_evaluation
            install_evaluation(engine)
    else:
        if args.rules != "legacy":
            parser.error("Copilot comparison uses the legacy MJAI bridge rules")
        from evaluation import copilot as engine
    sys.argv = [f"evaluation {args.method}", *remaining]
    engine.main()


if __name__ == "__main__":
    main()
