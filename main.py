from pathlib import Path
import argparse

import pygame

from gui.game_controller import GameController
from gui.game_window import MahjongGUI


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="三人日麻 GUI；未提供权重时使用随机合法动作 AI")
    parser.add_argument("--model", type=Path, default=project_dir / "model" / "model.pt")
    parser.add_argument("--replay-dir", type=Path, default=project_dir / "replays")
    args = parser.parse_args()
    gui = MahjongGUI(project_dir / "assets" / "tiles")
    controller = GameController(args.model, replay_dir=args.replay_dir)

    try:
        while gui.is_running():
            gui.handle_events(controller)
            controller.update(pygame.time.get_ticks())
            gui.render(controller)
    finally:
        gui.quit()


if __name__ == "__main__":
    main()
