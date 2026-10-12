"""Pause a training run and continue it exactly where it stopped.

A run folder (for example experiments/output/morph/v2) gets:

- ``launch.json``: how the run was started (python, script arguments, working folder, thread and path
  settings). The training script writes it at every start.
- ``PAUSE``: a pause request. The training loop checks it after every step. It then saves its complete
  state to ``checkpoints/paused.pt``, logs a "paused" event and exits. The state covers model and critic
  weights, optimizers, learning-rate schedule, every random generator, the running epoch's step records
  and the position.
- Resume: the same command with ``--resume``. The script loads the newer of ``paused.pt`` and
  ``last.pt`` and continues at the next step, with the same data order and random draws as if it had never
  stopped.

    python experiments/training_control.py pause  experiments/output/morph/v2
    python experiments/training_control.py resume experiments/output/morph/v2
    python experiments/training_control.py status experiments/output/morph/v2

The training dashboard has the same Pause / Resume buttons.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PAUSE_FILE = "PAUSE"
LAUNCH_FILE = "launch.json"
ENV_KEYS = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
STATE_FILES = ("paused.pt", "last.pt")


def record_launch(folder) -> None:
    """Called by a training script at start: remember how to start it again."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    launch = {"python": sys.executable, "argv": [a for a in sys.argv if a != "--resume"], "cwd": os.getcwd(),
              "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ}, "time": time.time()}
    (folder / LAUNCH_FILE).write_text(json.dumps(launch, indent=1), encoding="utf-8")


def request_pause(folder) -> None:
    (Path(folder) / PAUSE_FILE).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def pause_requested(folder) -> bool:
    return (Path(folder) / PAUSE_FILE).exists()


def clear_pause(folder) -> None:
    (Path(folder) / PAUSE_FILE).unlink(missing_ok=True)


def latest_state(checkpoints) -> Path | None:
    """The newer of ``paused.pt`` and ``last.pt`` (both hold a complete resumable state)."""
    found = [Path(checkpoints) / n for n in STATE_FILES if (Path(checkpoints) / n).exists()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def save_state(state: dict, path) -> None:
    """Write atomically (a stopped write never replaces a good state file)."""
    import torch
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def random_states(rng, generator) -> dict:
    """numpy Generator, torch Generator (augmentation) and torch's global CPU generator (dropout, drop path)."""
    import torch
    out = {"rng": rng.bit_generator.state, "torch_global": torch.get_rng_state()}
    if generator is not None:
        out["torch_generator"] = generator.get_state()
    return out


def restore_random(saved: dict, rng, generator) -> None:
    import torch
    rng.bit_generator.state = saved["rng"]
    if generator is not None and "torch_generator" in saved:
        generator.set_state(saved["torch_generator"])
    if "torch_global" in saved:
        torch.set_rng_state(saved["torch_global"])


def running_pid(folder) -> int | None:
    """The run's process id if it is alive and a Python process running a training script."""
    pid_file = Path(folder) / "train_pid.txt"
    text = pid_file.read_text().strip() if pid_file.exists() else ""
    if not text.isdigit():
        return None
    pid = int(text)
    if sys.platform == "win32":
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"],
                             capture_output=True, text=True, timeout=20).stdout
    else:
        cmdline = Path(f"/proc/{pid}/cmdline")
        out = cmdline.read_text().replace("\0", " ") if cmdline.exists() else ""
    return pid if "train_" in out else None


def resume(folder) -> int:
    """Start the run again with ``--resume`` (detached); returns the new process id."""
    folder = Path(folder).resolve()
    if running_pid(folder):
        raise RuntimeError("the run is still running")
    launch = json.loads((folder / LAUNCH_FILE).read_text(encoding="utf-8"))
    if latest_state(folder / "checkpoints") is None:
        raise RuntimeError("no paused.pt or last.pt to resume from")
    clear_pause(folder)
    env = dict(os.environ)
    env.update(launch.get("env", {}))
    stamp = time.strftime("%Y%m%d-%H%M")
    log = open(folder / f"train_log_{stamp}.txt", "a", encoding="utf-8")
    err = open(folder / "train_err.txt", "a", encoding="utf-8")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000           # no console window
    process = subprocess.Popen([launch["python"], *launch["argv"], "--resume"], cwd=launch["cwd"], env=env,
                               stdout=log, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags,
                               close_fds=True)
    (folder / "train_pid.txt").write_text(str(process.pid), encoding="ascii")
    return process.pid


def status(folder) -> dict:
    folder = Path(folder)
    state = latest_state(folder / "checkpoints")
    return {"running_pid": running_pid(folder), "pause_requested": pause_requested(folder),
            "resume_from": state.name if state else None,
            "resume_from_saved": time.strftime("%Y-%m-%d %H:%M", time.localtime(state.stat().st_mtime)) if state else None,
            "launch": json.loads((folder / LAUNCH_FILE).read_text(encoding="utf-8")) if (folder / LAUNCH_FILE).exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("pause", "resume", "status"))
    parser.add_argument("run", type=Path, help="run folder, e.g. experiments/output/morph/v2")
    args = parser.parse_args()
    if args.action == "pause":
        if not running_pid(args.run):
            raise SystemExit("not running")
        request_pause(args.run)
        print("pause requested: the run saves checkpoints/paused.pt after its current step and exits")
    elif args.action == "resume":
        print(f"resumed, PID {resume(args.run)}")
    else:
        print(json.dumps(status(args.run), indent=1, default=str))


if __name__ == "__main__":
    main()
