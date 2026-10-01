"""Install or launch the local Mahjong Copilot integration."""
from pathlib import Path
import os
import runpy
import subprocess
import sys
import venv

PROJECT_ROOT = Path(__file__).resolve().parent
COPILOT_ROOT = PROJECT_ROOT / "integrations" / "MahjongCopilot"
VENV_DIR = PROJECT_ROOT / ".venv-copilot"


def install_run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def setup_runtime() -> None:
    python_path = (
        VENV_DIR / "Scripts" / "python.exe"
        if os.name == "nt"
        else VENV_DIR / "bin" / "python"
    )
    if not python_path.is_file():
        print(f"Creating isolated runtime: {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    python = str(python_path)
    install_run(python, "-m", "pip", "install", "-U", "pip")
    install_run(python, "-m", "pip", "install", "torch", "numpy")
    install_run(python, "-m", "pip", "install", "-r", "requirements/copilot.txt")
    install_run(python, "-m", "playwright", "install", "chromium")
    # Fail during setup instead of later in the GUI if the browser archive was
    # interrupted or unpacked incorrectly.
    install_run(
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
        r".\.venv-copilot\Scripts\python.exe copilot.py"
        if os.name == "nt"
        else "./.venv-copilot/bin/python copilot.py"
    )


def ensure_playwright_browser() -> None:
    """Ensure Chromium is installed for the same Python that starts the app."""
    try:
        import playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run: python copilot.py --setup"
        ) from exc

    # Playwright knows the platform/architecture-specific executable location
    # (Windows, Intel macOS, Apple Silicon macOS and Linux). Avoid hardcoding
    # chrome-win64/chrome.exe here.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        executable = Path(runtime.chromium.executable_path)
    if executable.is_file():
        return

    print(f"Playwright Chromium is missing: {executable}", flush=True)
    print("Downloading the matching Chromium build...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not executable.is_file():
        raise RuntimeError(
            f"Chromium installation finished, but the executable is still missing: {executable}"
        )


def launch_copilot() -> None:
    if not (COPILOT_ROOT / "main.py").is_file():
        raise FileNotFoundError(f"MahjongCopilot is missing: {COPILOT_ROOT}")
    ensure_playwright_browser()
    os.chdir(COPILOT_ROOT)
    sys.path.insert(0, str(COPILOT_ROOT))
    runpy.run_path(str(COPILOT_ROOT / "main.py"), run_name="__main__")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Install or launch the local Mahjong Copilot integration")
    parser.add_argument("--setup", action="store_true", help="Create the isolated environment and install Chromium")
    args = parser.parse_args()
    if args.setup:
        setup_runtime()
    else:
        launch_copilot()


if __name__ == "__main__":
    main()
