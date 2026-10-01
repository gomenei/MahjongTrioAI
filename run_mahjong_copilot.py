"""Launch the vendored MahjongCopilot integration from the correct directory."""

from pathlib import Path
import os
import runpy
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
COPILOT_ROOT = PROJECT_ROOT / "integrations" / "MahjongCopilot"


def ensure_playwright_browser() -> None:
    """Ensure Chromium is installed for the same Python that starts the app."""
    try:
        import playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run: python setup_mahjong_copilot.py"
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


def main() -> None:
    if not (COPILOT_ROOT / "main.py").is_file():
        raise FileNotFoundError(f"MahjongCopilot is missing: {COPILOT_ROOT}")
    ensure_playwright_browser()
    os.chdir(COPILOT_ROOT)
    sys.path.insert(0, str(COPILOT_ROOT))
    runpy.run_path(str(COPILOT_ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
