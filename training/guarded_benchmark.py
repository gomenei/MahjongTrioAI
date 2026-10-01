"""Temporarily suspend this campaign for an uncontended, timeout-bounded benchmark."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-pid", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = psutil.Process(args.campaign_pid)
    if "run_autodl_campaign.py" not in " ".join(root.cmdline()):
        raise SystemExit("PID is not the expected campaign")
    frozen = []
    watchdog = None
    started = time.time()
    try:
        root.suspend()
        frozen.append(root)
        for process in root.children(recursive=True):
            try:
                process.suspend()
                frozen.append(process)
            except psutil.NoSuchProcess:
                pass
        identities = [(process.pid, process.create_time()) for process in frozen]
        rescue = ("import psutil,time\ntime.sleep(" + str(args.timeout + 30) + ")\n"
                  "for pid,born in " + repr(identities) + ":\n"
                  " try:\n  p=psutil.Process(pid)\n"
                  "  if p.create_time()==born: p.resume()\n"
                  " except psutil.NoSuchProcess: pass\n")
        watchdog = subprocess.Popen([sys.executable, "-c", rescue], start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(json.dumps({"suspended": [p.pid for p in frozen]}), flush=True)
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
        subprocess.run(command, check=True, timeout=args.timeout)
    finally:
        for process in reversed(frozen):
            try:
                process.resume()
            except psutil.NoSuchProcess:
                pass
        if watchdog is not None:
            watchdog.terminate()
            watchdog.wait()
        print(json.dumps({"resumed": True, "paused_seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
