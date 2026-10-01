"""Create an isolated MahjongCopilot runtime without changing the training env."""

from pathlib import Path
import os
import subprocess
import venv


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv-copilot"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    python_path = (
        VENV_DIR / "Scripts" / "python.exe"
        if os.name == "nt"
        else VENV_DIR / "bin" / "python"
    )
    if not python_path.is_file():
        print(f"Creating isolated runtime: {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    python = str(python_path)
    run(python, "-m", "pip", "install", "-U", "pip")
    run(python, "-m", "pip", "install", "torch", "numpy")
    run(python, "-m", "pip", "install", "-r", "requirements-copilot.txt")
    run(python, "-m", "playwright", "install", "chromium")
    # Fail during setup instead of later in the GUI if the browser archive was
    # interrupted or unpacked incorrectly.
    run(
        python,
        "-c",
        (
            "from playwright.sync_api import sync_playwright; "
            "p=sync_playwright().start(); "
            "b=p.chromium.launch(headless=True); "
            "v=b.version; "
            "b.close(); "
            "p.stop(); "
            "print(f'Chromium launch verified: {v}')"
        ),
    )
    print("\nSetup complete. Start with:")
    print(
        r".\.venv-copilot\Scripts\python.exe run_mahjong_copilot.py"
        if os.name == "nt"
        else "./.venv-copilot/bin/python run_mahjong_copilot.py"
    )


if __name__ == "__main__":
    main()
