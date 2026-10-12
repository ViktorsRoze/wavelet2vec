"""Training dashboard: a local, TensorBoard-like page for the training runs (KeyboardNet, MorphNet).

Serves http://localhost:6006 (standard library only, no extra packages).
A run is any folder under experiments/output/keyboard (KeyboardNet; the
keyboard folder itself is v1) or experiments/output/morph (MorphNet) that has
training output. The header picks the model system and then the version or
experiment; a live (training) run is listed first, opened by default and
marked with a green LIVE pill. For each run the page shows:

- status (running / stopped, from train_pid.txt), progress and ETA;
- curves: training loss, validation vs the pitch-shift and blend baselines,
  learning rate, gradient norms and update/weight ratios per module group,
  block update ratios and attention entropy per stage, probe scores;
- checkpoints with their metrics;
- evaluation reports (evals/*.json, *report*.json): accuracy, ablations, head
  and backbone diagnostics, probes;
- the end of the training log.

Data comes from metrics.jsonl (written by train_keyboard_v2.py). Older runs
are read from history.json and the train_log*.txt lines. The page refreshes
every 30 s; the server only reads files.

The encoding section (runs with encoding_examples.json, written by
experiments/encoding_examples.py) also draws pictures on demand, at the
canvas's own pixel size, from the exact wavelet image of each example note
(wavelet2vec.wavelet_image.detail_view / stereo_view): a zoomable viewer
(reassigned, exact-sample or cell-grid view; mid, side, left, right; stereo
pan and phase correlation) and a goniometer per note. Those two endpoints
need numpy, torch and wavelet2vec (the "sound" env); the rest of the page
does not.

    python experiments/training_dashboard.py [--port 6006]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import training_control as ctl

HERE = Path(__file__).resolve().parent
ROOT = HERE / "output" / "keyboard"
MORPH = HERE / "output" / "morph"
AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac"}
LOG_LINE = re.compile(r"epoch\s+(\d+) step\s+(\d+)/(\d+)\s+env\s+([\d.]+) dB\s+aux\s+([\d.]+)\s+lr\s+([\d.e+-]+)")
OLD_CKPT = re.compile(r"epoch_(\d+)\.pt$")
NEW_CKPT = re.compile(r"([\d.]+)_epoch\.pth$")


def discover_runs():
    runs = {}
    candidates = [ROOT] + [p for p in sorted(ROOT.iterdir()) if p.is_dir()] if ROOT.exists() else []
    for folder in candidates:
        has = any((folder / n).exists() for n in ("metrics.jsonl", "history.json")) or list(folder.glob("train_log*.txt")) \
            or list(folder.glob("*report*.json"))
        if has:
            runs["v1" if folder == ROOT else folder.name] = folder
    if MORPH.exists():
        for folder in sorted(MORPH.iterdir()):
            if folder.is_dir() and (folder / "metrics.jsonl").exists():
                runs[f"morph/{folder.name}"] = folder
    return runs


FAMILIES = (("morph/", "MorphNet"), ("", "KeyboardNet"))


_TRAINING = {"time": 0.0, "pids": set()}


def training_pids():
    """Process ids of Python processes running a training script (``train_*.py``) now; cached for 10 s.
    Checking the command line, not just the id, avoids reused process ids marking old runs as live."""
    if time.time() - _TRAINING["time"] < 10:
        return _TRAINING["pids"]
    pids = set()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            pid, _, cmd = line.partition("	")
            if pid.strip().isdigit() and "train_" in cmd:
                pids.add(int(pid))
    except Exception:
        pass
    _TRAINING.update(time=time.time(), pids=pids)
    return pids


def run_index():
    """Every run with its model system, version, live status and last update: live runs first, then newest."""
    alive = training_pids()
    out = []
    for name, folder in discover_runs().items():
        family, version = next((f, name[len(prefix):]) for prefix, f in FAMILIES if name.startswith(prefix))
        pid_file = folder / "train_pid.txt"
        text = pid_file.read_text().strip() if pid_file.exists() else ""
        pid = int(text) if text.isdigit() else None
        files = [folder / "metrics.jsonl", folder / "history.json", *folder.glob("train_log*.txt")]
        updated = max((f.stat().st_mtime for f in files if f.exists()), default=0.0)
        out.append({"name": name, "family": family, "version": version, "pid": pid,
                    "running": bool(pid) and pid in alive, "updated": updated})
    out.sort(key=lambda r: (not r["running"], -r["updated"]))
    return out


def process_running(pid):
    return pid in training_pids()


def read_events(folder):
    events = []
    path = folder / "metrics.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    covered = {round(e["epoch"], 3) for e in events if e["type"] == "val"}
    history = folder / "history.json"
    if history.exists():  # older runs: one validation per whole epoch
        for record in json.loads(history.read_text(encoding="utf-8")):
            if round(float(record["epoch"]), 3) not in covered:
                events.append({"type": "val", "epoch": float(record["epoch"]), "val": record["val"],
                               "train_env_db": record.get("train_env_db"), "legacy": True})
    if not any(e["type"] == "train" for e in events):
        for log in sorted(folder.glob("train_log*.txt")):
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                m = LOG_LINE.search(line)
                if m:
                    ep, step, total = int(m[1]), int(m[2]), int(m[3])
                    events.append({"type": "train", "epoch": ep - 1 + step / total, "env_db": float(m[4]),
                                   "aux": float(m[5]), "lr": float(m[6]), "legacy": True})
        v1_log = folder / "train_log.txt"
        if not events and v1_log.exists():
            for line in v1_log.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.search(r"step\s+(\d+)\s+loss\s+([\d.]+) dB", line)
                if m:
                    events.append({"type": "train", "epoch": int(m[1]) / 4000, "env_db": float(m[2]), "legacy": True})
    events.sort(key=lambda e: (e.get("epoch", 0), e.get("time", 0)))
    return events


def _val_score(val):
    """The headline validation number of a val event, whatever the run type (None if there is none)."""
    if not val:
        return None
    if isinstance(val.get("val"), dict) and "model" in val["val"]:
        return val["val"]["model"]
    if isinstance(val.get("morph"), dict):
        return val["morph"].get("score")
    return val.get("score")


def checkpoints(folder, events):
    by_epoch = {round(e["epoch"], 3): e for e in events if e["type"] == "val"}
    best = None
    for e in events:
        if e["type"] == "checkpoint" and e.get("best"):
            best = e["file"]
    rows = []
    for path in sorted((folder / "checkpoints").glob("*")) if (folder / "checkpoints").exists() else []:
        if path.suffix not in (".pt", ".pth"):
            continue
        m = NEW_CKPT.search(path.name) or OLD_CKPT.search(path.name)
        epoch = float(m[1]) if m else None
        val = by_epoch.get(round(epoch, 3)) if epoch is not None else None
        stat = path.stat()
        rows.append({"file": path.name, "epoch": epoch, "size_mb": round(stat.st_size / 1e6, 1),
                     "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                     "val_model": _val_score(val),
                     "best": path.name == best})
    if best is None:  # older runs have no checkpoint events: mark the lowest validation error
        scored = [r for r in rows if r["val_model"] is not None]
        if scored:
            min(scored, key=lambda r: r["val_model"])["best"] = True
    rows.sort(key=lambda r: (r["epoch"] is None, r["epoch"] or 0, r["file"]))
    return rows


def evaluations(folder):
    found = []
    for path in sorted(list(folder.glob("*report*.json")) + list((folder / "evals").glob("*.json"))):
        try:
            found.append({"file": path.name, "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                          "report": json.loads(path.read_text(encoding="utf-8"))})
        except (OSError, json.JSONDecodeError):
            pass
    return found


def log_tail(folder, lines=200):
    logs = sorted(folder.glob("train_log*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return ""
    text = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    err = folder / "train_err.txt"
    errors = []
    if err.exists():
        errors = [l for l in err.read_text(encoding="utf-8", errors="replace").splitlines()
                  if l.strip() and "Warning" not in l and "warn(" not in l and "Consider using" not in l]
    tail = f"[{logs[-1].name}]\n" + "\n".join(text[-lines:])
    if errors:
        tail += "\n\n[train_err.txt]\n" + "\n".join(errors[-30:])
    return tail


class EncodingViews:
    """Pictures of the encoding examples at screen resolution, computed on request and cached per note."""

    MODES = ("sharp", "exact", "cells")
    SHOWS = ("level", "pan", "corr")
    RANGE_DB = 100.0

    def __init__(self):
        self.lock = threading.Lock()
        self.busy = threading.Semaphore(2)
        self.images, self.audio, self.refs, self.stereo, self.demos = {}, {}, {}, {}, {}

    def _libs(self):
        src = str(HERE.parent / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        import numpy as np
        import torch
        from wavelet2vec import wavelet_image
        torch.set_num_threads(2)
        return np, torch, wavelet_image

    def _audio(self, folder):
        np, _, _ = self._libs()
        key = str(folder)
        with self.lock:
            if key not in self.audio:
                self.audio[key] = np.load(folder / "encoding_examples_audio.npy", mmap_mode="r")
            return self.audio[key]

    def _image(self, folder, ex, source):
        np, torch, wi = self._libs()
        key = (str(folder), ex, source)
        with self.lock:
            if key in self.images:
                return self.images[key]
        audio = torch.from_numpy(np.asarray(self._audio(folder)[ex], dtype=np.float64))[None]
        from wavelet2vec.keyboard_v3 import SPEC, shift_image
        image = wi.encode(audio, SPEC)
        if source == "carrier":
            meta = json.loads((folder / "encoding_examples.json").read_text(encoding="utf-8"))
            image = shift_image(image, [meta.get("shift_semitones", 5)])
        with self.lock:
            self.images[key] = image
        return image

    def _compute(self, key, image, mode, show, channel, **frame):
        _, _, wi = self._libs()
        if show != "level":
            key = key + (mode,) + tuple(sorted((k, str(v)) for k, v in frame.items()))
            if key not in self.stereo:
                if len(self.stereo) > 12:
                    self.stereo.pop(next(iter(self.stereo)))
                self.stereo[key] = wi.stereo_view(image, mode=mode, **frame)
            level, balance, correlation = self.stereo[key]
            return level, balance if show == "pan" else correlation
        return wi.detail_view(image, mode=mode, channel=channel, **frame), None

    def view(self, folder, q):
        """uint8 bytes: level (0..254 = -100..0 dB re the loudest pixel of the whole note, 255 = quieter);
        for show=pan/corr, a second plane follows with the value (-1..1 -> 0..254). mode = picture:
        sharp (reassigned), exact (every sample at its own rate) or cells (the backbone's grid)."""
        np, _, _ = self._libs()
        ex, source = int(q.get("ex", 0)), q.get("src", "note")
        mode, show, channel = q.get("mode", "sharp"), q.get("show", "level"), q.get("ch", "mid")
        if mode not in self.MODES or show not in self.SHOWS or source not in ("note", "carrier"):
            raise ValueError("unknown mode, show or source")
        width = max(16, min(2400, int(q.get("w", 800))))
        height = max(16, min(1400, int(q.get("h", 400))))
        t0, t1 = float(q.get("t0", 0.0)), float(q.get("t1", 1.2))
        m0, m1 = float(q.get("m0", 11.5)), float(q.get("m1", 160.5))
        whole = (t0, t1, m0, m1) == (0.0, 1.2, 11.5, 160.5)
        with self.busy:
            image = self._image(folder, ex, source)
            key = (str(folder), ex, source)
            ref_key = key + (mode, "stereo" if show != "level" else channel)
            if ref_key not in self.refs and not whole:
                full, _ = self._compute(key, image, mode, show, channel, width=480, height=298)
                self.refs[ref_key] = float(full.max())
            level, value = self._compute(key, image, mode, show, channel, time_range=(t0, t1),
                                         midi_range=(m0, m1), width=width, height=height)
            if ref_key not in self.refs:
                self.refs[ref_key] = float(level.max())
        lo = self.refs[ref_key] - self.RANGE_DB
        out = np.clip(np.round((level - lo) / self.RANGE_DB * 254), 0, 254).astype(np.uint8)
        out[(level <= lo) | (level <= -139.5)] = 255          # -140 dB is the views' floor: exactly silent
        planes = [out]
        if value is not None:
            planes.append(np.clip(np.round((value + 1) / 2 * 254), 0, 254).astype(np.uint8))
        return b"".join(p.tobytes() for p in planes)

    def demo(self, folder, q):
        """Spectrogram of one audio file of a run (listening demos): uint8 levels, 0..254 = -90..0 dB re the
        file's loudest pixel, 255 = quieter; row 0 = lowest pitch. Drawn from the file's exact wavelet image."""
        np, torch, wi = self._libs()
        import soundfile as sf
        from wavelet2vec.keyboard_v3 import SPEC
        folder = Path(folder).resolve()
        path = (folder / q.get("path", "")).resolve()
        if folder not in path.parents or path.suffix.lower() not in AUDIO_TYPES or not path.is_file():
            raise ValueError("no such audio file in this run")
        mode = q.get("mode", "sharp")
        width = max(16, min(1600, int(q.get("w", 300))))
        height = max(16, min(800, int(q.get("h", 120))))
        m0, m1 = float(q.get("m0", 21.0)), float(q.get("m1", 136.0))
        key = (str(path), path.stat().st_mtime, mode, width, height, m0, m1)
        with self.lock:
            hit = self.demos.get(key)
        if hit is not None:
            return hit
        with self.busy:
            audio, rate = sf.read(path, dtype="float64", always_2d=True)
            if rate != SPEC.sample_rate:
                raise ValueError("demo files are 192 kHz")
            x = np.zeros((1, 2, SPEC.audio_length))
            n = min(len(audio), SPEC.audio_length)
            x[0, :, :n] = audio[:n, :2].T if audio.shape[1] > 1 else np.repeat(audio[:n, :1].T, 2, 0)
            image = wi.encode(torch.from_numpy(x), SPEC)
            level = wi.detail_view(image, mode=mode, channel="mid", midi_range=(m0, m1), width=width, height=height)
        lo = float(level.max()) - 90.0
        out = np.clip(np.round((level - lo) / 90.0 * 254), 0, 254).astype(np.uint8)
        out[level <= max(lo, -139.5)] = 255
        data = out.tobytes()
        with self.lock:
            if len(self.demos) > 400:
                self.demos.pop(next(iter(self.demos)))
            self.demos[key] = data
        return data

    def scope(self, folder, q):
        """Goniometer density, uint8 [size, size] (row 0 = top): x = (R - L)/sqrt 2, y = (L + R)/sqrt 2."""
        np, _, _ = self._libs()
        ex, size = int(q.get("ex", 0)), max(32, min(1400, int(q.get("size", 300))))
        audio = np.asarray(self._audio(folder)[ex], dtype=np.float64)
        x, y = (audio[1] - audio[0]) / np.sqrt(2), (audio[0] + audio[1]) / np.sqrt(2)
        scale = max(float(np.abs(x).max()), float(np.abs(y).max()), 1e-12) * 1.04
        counts, _, _ = np.histogram2d(-y / scale, x / scale, bins=size, range=[[-1, 1], [-1, 1]])
        nonzero = counts[counts > 0]
        top = np.log1p(np.percentile(nonzero, 99.5)) if nonzero.size else 1.0
        density = np.clip(np.log1p(counts) / max(top, 1e-9), 0, 1)
        out = np.round(density * 254).astype(np.uint8)
        out[counts == 0] = 255
        return out.tobytes()


VIEWS = EncodingViews()


def run_payload(name, folder):
    events = read_events(folder)
    pid_file = folder / "train_pid.txt"
    pid = int(pid_file.read_text().strip()) if pid_file.exists() and pid_file.read_text().strip().isdigit() else None
    running = bool(pid) and process_running(pid)
    start = next((e for e in reversed(events) if e["type"] == "run_start"), None)
    train = [e for e in events if e["type"] == "train"]
    marks = [e for e in events if e["type"] in ("paused", "run_start")]
    last_mark = max(marks, key=lambda e: e.get("time", 0)) if marks else None
    state_file = ctl.latest_state(folder / "checkpoints")
    info = {"name": name, "folder": str(folder), "running": running, "pid": pid, "run": start,
            "pause_requested": ctl.pause_requested(folder),
            "paused": (not running) and bool(last_mark) and last_mark["type"] == "paused",
            "paused_at": last_mark.get("epoch") if last_mark and last_mark["type"] == "paused" else None,
            "can_resume": (not running) and (folder / ctl.LAUNCH_FILE).exists() and state_file is not None,
            "resume_from": state_file.name if state_file else None}
    if train:
        last = train[-1]
        info["epoch"] = round(last["epoch"], 3)
        speeds = [e["sec_per_step"] for e in train[-20:] if e.get("sec_per_step")]
        if start and speeds:
            steps_left = (start.get("epochs_total", 50) - last["epoch"]) * start["steps_per_epoch"]
            info["eta_hours"] = round(steps_left * sum(speeds) / len(speeds) / 3600, 1)
    def safe(fn, *a):                               # one broken section must not blank the page
        try:
            return fn(*a)
        except Exception as error:
            print(f"dashboard: {fn.__name__} failed for {name}: {error!r}", flush=True)
            return []
    return {"info": info, "events": events, "checkpoints": safe(checkpoints, folder, events),
            "evaluations": safe(evaluations, folder), "log": log_tail(folder)}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Training dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--accent:#2563eb;--good:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--panel:#1c1c1c;--ink:#ececea;--muted:#9a9a94;--line:#2e2e2b;--accent:#60a5fa;--good:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0 8px 0 0}select,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
.badge{padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.badge.live{background:var(--good);border-color:var(--good);color:var(--panel);font-weight:600;padding:3px 10px}
.badge.live::before,.golive::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:currentColor;margin-right:6px;vertical-align:1px;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.golive{border-color:var(--good)!important;color:var(--good)!important;font-weight:600;cursor:pointer}
.hsel{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
.badge.paused{background:#d97706;border-color:#d97706;color:var(--panel);font-weight:600;padding:3px 10px}
.ctlbtn{font-weight:600;cursor:pointer}.ctlbtn.pause{border-color:#d97706;color:#d97706}.ctlbtn.resume{border-color:var(--good);color:var(--good)}
.ctlbtn:disabled{opacity:.6;cursor:default}.run{color:var(--good)}.stop{color:var(--muted)}
main{padding:16px 20px;display:grid;gap:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}h2{font-size:14px;margin:0 0 8px}
canvas{width:100%;height:240px;display:block}.legend{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--muted);margin-top:4px}
.legend i{display:inline-block;width:10px;height:3px;margin-right:4px;vertical-align:middle}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}.scroll{overflow:auto;max-height:420px}pre{margin:0;font:12px/1.4 ui-monospace,Consolas,monospace;white-space:pre-wrap;max-height:420px;overflow:auto}
.muted{color:var(--muted)}.best{color:var(--good);font-weight:600}.kv{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px}.kv b{font-weight:600}
@media (max-width:520px){.grid{grid-template-columns:1fr}main{padding:12px}}
.enc3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.encrow{margin-top:14px}
.encrow h3{font-size:13px;margin:0 0 2px}.encrow p{margin:0 0 6px;font-size:12px;color:var(--muted)}
.card{border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12.5px}.card b{font-size:13px}
.hm{width:100%;height:190px;display:block;cursor:crosshair}.wave{width:100%;height:90px;display:block}
.scale{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted);margin-top:4px}.scale canvas{width:160px;height:10px;border-radius:2px}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;display:none;z-index:5;box-shadow:0 2px 8px rgba(0,0,0,.15)}
@media (max-width:900px){.enc3{grid-template-columns:1fr}}
.dv{width:100%;height:480px;display:block;cursor:crosshair;user-select:none}.demospec{width:190px;height:96px;display:block;margin-top:4px;border-radius:3px;cursor:crosshair}.gonio{width:100%;height:250px;display:block}
.dvbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0 8px;font-size:12px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:3px 9px;font-size:12px;cursor:pointer}.seg button+button{border-left:1px solid var(--line)}
.seg button.on{background:var(--accent);color:var(--panel)}
.meter{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted);margin:6px 0 2px}
.meter .bar{position:relative;flex:1;height:8px;border-radius:4px}
.meter .bar i{position:absolute;top:-3px;width:3px;height:14px;margin-left:-1.5px;background:var(--ink);border-radius:1px;box-shadow:0 0 0 2px var(--panel)}
</style></head><body>
<header><h1>Training dashboard</h1><label class="hsel">model system <select id="family"></select></label>
<label class="hsel">version / experiment <select id="run"></select></label><span id="status" class="badge"></span>
<button id="golive" class="golive" style="display:none"></button>
<button id="pausebtn" class="ctlbtn pause" style="display:none">Pause</button><button id="resumebtn" class="ctlbtn resume" style="display:none">Resume training</button>
<span id="progress" class="muted"></span><button id="refresh">Refresh</button><span class="muted" id="updated"></span></header>
<main>
<section><h2>Run</h2><div class="kv" id="runinfo"></div></section>
<section id="encSection" style="display:none"><h2>Encoding examples <span class="muted">(three unrelated notes, three source resolutions, through the run's encoding)</span></h2>
<div id="encoding"></div></section><div id="tip"></div>
<div class="grid" id="charts"></div>
<section id="morphSection" style="display:none"><h2>Vox–synth morph <span class="muted">(judges, listening)</span></h2><div id="morph"></div></section>
<section><h2>Probes over training <span class="muted">(ridge probes on pooled features, scored on held-out families)</span></h2>
<div class="kv"><label>feature <select id="probeFeature"></select></label></div><div class="grid" id="probeCharts"></div></section>
<section><h2>Checkpoints</h2><div class="scroll"><table id="ckpts"></table></div></section>
<section><h2>Evaluations</h2><select id="evalSel"></select><div id="evalBody"></div></section>
<section><h2>Training log (tail)</h2><pre id="log"></pre></section>
</main>
<script>
const COLORS=["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#0891b2","#db2777","#65a30d","#475569"];
let state={run:null,data:null};
const $=id=>document.getElementById(id);
async function getJSON(u){const r=await fetch(u);return r.json()}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function chart(title,series,opts={}){
  const sec=document.createElement("section");sec.innerHTML=`<h2>${title}</h2><canvas></canvas><div class="legend"></div>`;
  const cv=sec.querySelector("canvas"),lg=sec.querySelector(".legend");
  series=series.filter(s=>s.points.length);
  lg.innerHTML=series.map((s,i)=>`<span><i style="background:${s.color||COLORS[i%COLORS.length]}"></i>${s.name}</span>`).join("");
  requestAnimationFrame(()=>draw(cv,series,opts));return sec;
}
function draw(cv,series,opts){
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*dpr;cv.height=h*dpr;
  const g=cv.getContext("2d");g.scale(dpr,dpr);g.font="11px system-ui";const ink=css("--muted"),line=css("--line");
  const pts=series.flatMap(s=>s.points).filter(p=>p[1]!=null&&isFinite(p[1])&&(!opts.log||p[1]>0));
  if(!pts.length){g.fillStyle=ink;g.fillText("no data yet",10,20);return}
  const tf=v=>opts.log?Math.log10(v):v;let x0=Math.min(...pts.map(p=>p[0])),x1=Math.max(...pts.map(p=>p[0]));
  let y0=Math.min(...pts.map(p=>tf(p[1]))),y1=Math.max(...pts.map(p=>tf(p[1])));if(x0===x1){x1=x0+1}if(y0===y1){y0-=1;y1+=1}
  const pad=(y1-y0)*0.06;y0-=pad;y1+=pad;const L=48,R=8,T=8,B=22;
  const X=x=>L+(x-x0)/(x1-x0)*(w-L-R),Y=y=>T+(1-(tf(y)-y0)/(y1-y0))*(h-T-B);
  g.strokeStyle=line;g.fillStyle=ink;g.lineWidth=1;
  for(let i=0;i<=4;i++){const yv=y0+(y1-y0)*i/4,py=T+(1-i/4)*(h-T-B);g.beginPath();g.moveTo(L,py);g.lineTo(w-R,py);g.stroke();
    const lab=opts.log?Number(Math.pow(10,yv).toPrecision(2)).toExponential(0):(Math.abs(yv)<10?yv.toFixed(2):yv.toFixed(1));g.fillText(lab,2,py+4)}
  for(let i=0;i<=5;i++){const xv=x0+(x1-x0)*i/5;g.fillText(xv.toFixed(xv<10?1:0),X(xv)-8,h-6)}
  series.forEach((s,i)=>{g.strokeStyle=s.color||COLORS[i%COLORS.length];g.lineWidth=s.width||1.6;g.setLineDash(s.dash||[]);g.beginPath();let on=false;
    s.points.forEach(p=>{if(p[1]==null||!isFinite(p[1])||(opts.log&&p[1]<=0)){on=false;return}const px=X(p[0]),py=Y(p[1]);on?g.lineTo(px,py):g.moveTo(px,py);on=true});g.stroke();
    if(s.dots){g.fillStyle=g.strokeStyle;s.points.forEach(p=>{if(p[1]!=null){g.beginPath();g.arc(X(p[0]),Y(p[1]),2.5,0,7);g.fill()}})}});
  g.setLineDash([]);g.fillStyle=ink;g.fillText(opts.xlabel||"epoch",w-R-34,h-6);
}
function smooth(points,k){if(points.length<k)return points;return points.map((p,i)=>{const a=points.slice(Math.max(0,i-k+1),i+1);return[p[0],a.reduce((s,q)=>s+q[1],0)/a.length]})}
function series(evs,key,getter){return evs.filter(e=>e.type===key).map(e=>[e.epoch,getter(e)])}
function groupSeries(evs,field){const names=new Set();evs.filter(e=>e.type==="train"&&e[field]).forEach(e=>Object.keys(e[field]).forEach(k=>names.add(k)));
  return[...names].map(n=>({name:n,points:series(evs,"train",e=>e[field]?e[field][n]:null)}))}
function render(){
  const d=state.data,ev=d.events,info=d.info;
  $("status").textContent=info.running?`LIVE training · PID ${info.pid}`:info.paused?`PAUSED at epoch ${(+info.paused_at).toFixed(3)}`:"stopped";
  $("status").className="badge "+(info.running?"live":info.paused?"paused":"stop");
  const pb=$("pausebtn"),rb=$("resumebtn");pb.style.display=info.running?"":"none";pb.disabled=!!info.pause_requested;
  pb.textContent=info.pause_requested?"Pausing after this step…":"Pause";rb.style.display=info.can_resume?"":"none";
  rb.textContent=info.paused?"Resume training (exactly where it paused)":`Resume from ${info.resume_from||"checkpoint"}`;
  $("progress").textContent=info.epoch!=null?`epoch ${info.epoch}`+(info.running&&info.eta_hours!=null?` · ETA ${info.eta_hours} h`:""):"";
  const r=info.run||{};$("runinfo").innerHTML=Object.entries({"model size":r.size,"parameters":r.parameters?(r.parameters.total/1e6).toFixed(1)+"M":null,
    "batch":r.batch,"steps/epoch":r.steps_per_epoch,"train notes":r.train_notes&&typeof r.train_notes==="object"?Object.entries(r.train_notes).map(([k,v])=>`${k} ${v}`).join(", "):r.train_notes,"val families":(r.val_families||[]).join(", "),
    "test families":(r.test_families||[]).length||null,"folder":info.folder}).filter(([k,v])=>v!=null&&v!=="").map(([k,v])=>`<span><b>${k}</b> ${v}</span>`).join("")||"<span class=muted>no run_start event (older run)</span>";
  const C=$("charts");C.innerHTML="";
  const morph=r.task==="morph"||r.task==="diffusion";$("morphSection").style.display=morph?"":"none";
  if(morph){(r.task==="diffusion"?renderDiffusion:renderMorph)(C,ev);renderProbes();renderCkpts();renderEvals();$("log").textContent=d.log||"(no log)";$("log").scrollTop=1e9;
    $("updated").textContent="updated "+new Date().toLocaleTimeString();return}
  const tr=series(ev,"train",e=>e.env_db);
  C.append(chart("Training loss: envelope error (dB)",[{name:"per log step",points:tr,color:"#94a3b8",width:1},{name:"smoothed",points:smooth(tr,8),color:COLORS[0],width:2}]));
  const vals=ev.filter(e=>e.type==="val");
  C.append(chart("Validation: envelope error (dB)",[{name:"model",points:vals.map(e=>[e.epoch,e.val.model]),dots:true},
    {name:"pitch-shift nearest",points:vals.map(e=>[e.epoch,e.val.shift_nearest]),dash:[4,3]},{name:"blend of examples",points:vals.map(e=>[e.epoch,e.val.blend]),dash:[2,3]},
    {name:"train (epoch mean)",points:vals.filter(e=>e.train_env_db).map(e=>[e.epoch,e.train_env_db]),dash:[6,2]}]));
  const byk=[1,2,3].map(k=>({name:k+" example"+(k>1?"s":""),points:vals.filter(e=>e.val_by_k).map(e=>{const cells=Object.entries(e.val_by_k).filter(([n])=>n.startsWith(k+"-shot"));
    return[e.epoch,cells.length?cells.reduce((s,[,v])=>s+v.model,0)/cells.length:null]}),dots:true}));
  C.append(chart("Validation by number of given notes (dB)",byk));
  C.append(chart("Learning rate",[{name:"lr",points:series(ev,"train",e=>e.lr)}],{log:true}));
  C.append(chart("Gradient norm per module group",groupSeries(ev,"grad_norm"),{log:true}));
  C.append(chart("Update / weight ratio per step (healthy ≈ 1e-4 – 1e-2)",groupSeries(ev,"update_ratio"),{log:true}));
  const stages=["stage1","stage2","stage3","stage4"];
  C.append(chart("Block utilization: residual update / input (attention)",stages.map(s=>({name:s,points:series(ev,"train",e=>e.block_update&&e.block_update[s]?e.block_update[s].attn:null)})),{log:true}));
  C.append(chart("Block utilization: residual update / input (MLP)",stages.map(s=>({name:s,points:series(ev,"train",e=>e.block_update&&e.block_update[s]?e.block_update[s].mlp:null)})),{log:true}));
  C.append(chart("Attention entropy (1 = uniform, lower = focused)",stages.map(s=>({name:s,points:series(ev,"train",e=>e.attention_entropy?e.attention_entropy[s]:null)}))));
  renderProbes();renderCkpts();renderEvals();$("log").textContent=d.log||"(no log)";$("log").scrollTop=1e9;
  $("updated").textContent="updated "+new Date().toLocaleTimeString();
}
function flat(evs,type,get){return evs.filter(e=>e.type===type).map(e=>[e.epoch,get(e)]).filter(p=>p[1]!=null&&isFinite(p[1]))}
function refLine(name,points,y,color){return points.length?{name,points:[[points[0][0],y],[points[points.length-1][0],y]],dash:[3,4],color:color||"#94a3b8",width:1}:{name,points:[]}}
function renderMorph(C,ev){
  const L=k=>flat(ev,"train",e=>e.losses?e.losses[k]:null),V=(who,k)=>flat(ev,"val",e=>e.morph&&e.morph[who]?e.morph[who][k]:null);
  C.append(chart("Balance loss: smooth-L1 of (logit vox − logit synth) − logit(α), both judges",[{name:"balance",points:smooth(L("balance"),8),color:COLORS[0]}]));
  C.append(chart("Other generator objectives (training, smoothed; 0 = satisfied)",["novelty","between","rank","matching","articulation","anchor","residual","loudness","smooth"].map((k,i)=>({name:k,points:smooth(L(k),8),color:COLORS[(i+1)%COLORS.length]}))));
  if(L("vocality_A").length)C.append(chart("Vocality of the hybrids while training (judge A embedding axis: synth 0, vox 1)",[{name:"vocality (A)",points:smooth(L("vocality_A"),8),color:COLORS[0]},refLine("0.5",L("vocality_A"),0.5)]));
  const pv=L("p_vox_A");
  C.append(chart("Judge A: P(vox) of the hybrids while training (target α; 0.5 on half the pairs)",[{name:"per log step",points:pv,color:"#94a3b8",width:1},{name:"smoothed",points:smooth(pv,8),color:COLORS[0],width:2},refLine("0.5",pv,0.5)]));
  C.append(chart("Detectors on the hybrids (judge A): both should be high, not both low",[{name:"P(vox present)",points:smooth(L("vox_present_A"),8),color:COLORS[0]},
    {name:"P(synth present)",points:smooth(L("synth_present_A"),8),color:COLORS[1]},refLine("0.5",L("vox_present_A"),0.5)]));
  C.append(chart("Presence loss: softplus(2 − logit) for both detectors (lower = both present)",[{name:"presence",points:smooth(L("presence"),8),color:COLORS[2]}]));
  C.append(chart("Adversarial: generator realism term and critic hinge loss",[{name:"realism term (−critic score)",points:smooth(L("realism"),8),color:COLORS[3]},{name:"critic loss",points:smooth(L("critic"),8),color:COLORS[4]}]));
  const vh=V("hybrid","p_vox_A");
  C.append(chart("Validation: P(vox) by judge (held-out singers' takes × held-out synth families)",[
    {name:"hybrid, judge A",points:vh,dots:true,color:COLORS[0]},{name:"hybrid, judge B",points:V("hybrid","p_vox_B"),dots:true,color:COLORS[1]},
    {name:"hybrid, held-out judge J",points:V("hybrid","p_vox_J"),dots:true,color:COLORS[2]},
    {name:"plain mix, judge A",points:V("mix","p_vox_A"),dash:[4,3],color:COLORS[3]},{name:"vocoder (synth carrier), judge A",points:V("vocoder: synth carrier","p_vox_A"),dash:[2,3],color:COLORS[4]},refLine("target 0.5",vh,0.5)]));
  C.append(chart("Validation: balance error |P(vox) − α| (lower is better)",[{name:"judge A",points:V("hybrid","balance_error_A"),dots:true,color:COLORS[0]},
    {name:"judge B",points:V("hybrid","balance_error_B"),dots:true,color:COLORS[1]},{name:"held-out judge J",points:V("hybrid","balance_error_J"),dots:true,color:COLORS[2]}]));
  const dm=V("hybrid","distance_to_mix_A");
  C.append(chart("Validation novelty: hybrid's distance to the baselines (÷ vocal–synth distance, judge A embedding)",[
    {name:"to plain mix",points:dm,dots:true,color:COLORS[3]},{name:"to vocoder, synth carrier",points:V("hybrid","distance_to_vocoder: synth carrier_A"),dots:true,color:COLORS[4]},
    {name:"to vocoder, vocal carrier",points:V("hybrid","distance_to_vocoder: vocal carrier_A"),dots:true,color:COLORS[5]},refLine("required 0.25",dm,0.25,"#b91c1c")]));
  C.append(chart("Validation realism gap: critic score of real notes − generated (lower = more real)",[{name:"hybrid",points:V("hybrid","realism_gap"),dots:true,color:COLORS[0]},
    {name:"plain mix",points:V("mix","realism_gap"),dash:[4,3],color:COLORS[3]},{name:"vocoder (synth carrier)",points:V("vocoder: synth carrier","realism_gap"),dash:[2,3],color:COLORS[4]}]));
  if(V("hybrid","vocality_A").length){C.append(chart("Validation vocality (judge embedding axis; target α)",[{name:"hybrid, judge A",points:V("hybrid","vocality_A"),dots:true,color:COLORS[0]},
      {name:"hybrid, judge B",points:V("hybrid","vocality_B"),dots:true,color:COLORS[1]},{name:"plain mix, judge A",points:V("mix","vocality_A"),dash:[4,3],color:COLORS[3]},
      {name:"vocoder (synth carrier), judge A",points:V("vocoder: synth carrier","vocality_A"),dash:[2,3],color:COLORS[4]},refLine("0.5",V("hybrid","vocality_A"),0.5)]));
    C.append(chart("Validation: fraction of hybrids with both detectors on",[{name:"judge A",points:V("hybrid","both_present_A"),dots:true,color:COLORS[0]},{name:"judge B",points:V("hybrid","both_present_B"),dots:true,color:COLORS[1]}]));
    C.append(chart("Validation KID realism in Wavelet2Vec space (lower = closer to real notes)",[{name:"hybrid",points:V("hybrid","kid"),dots:true,color:COLORS[0]},
      {name:"plain mix",points:V("mix","kid"),dash:[4,3],color:COLORS[3]},{name:"vocoder (synth carrier)",points:V("vocoder: synth carrier","kid"),dash:[2,3],color:COLORS[4]}]))}
  if(V("hybrid","articulation_keep_acc").length)C.append(chart("Validation articulation: vowels judged as intended (share of vowel frames)",[
    {name:"kept: hybrid, synth pitch",points:V("hybrid","articulation_keep_acc"),dots:true,color:COLORS[0]},{name:"kept: hybrid, vocal pitch",points:V("hybrid, vocal pitch","articulation_keep_acc"),dots:true,color:COLORS[1]},
    {name:"controlled: other vowel requested",points:V("hybrid","articulation_control_acc"),dots:true,color:COLORS[2]},{name:"the voice itself (ceiling)",points:V("vocal","articulation_keep_acc"),dash:[4,3],color:COLORS[4]}]));
  if(V("hybrid","pitch_jitter_cents").length)C.append(chart("Validation pitch stability: spread of harmonics 1–3 in cents (lower = steadier)",[
    {name:"hybrid, synth pitch",points:V("hybrid","pitch_jitter_cents"),dots:true,color:COLORS[0]},{name:"hybrid, vocal pitch",points:V("hybrid, vocal pitch","pitch_jitter_cents"),dots:true,color:COLORS[1]},
    {name:"synth (source)",points:V("synth","pitch_jitter_cents"),dash:[4,3],color:COLORS[2]},{name:"vocal (source)",points:V("vocal","pitch_jitter_cents"),dash:[2,3],color:COLORS[4]},
    {name:"plain mix",points:V("mix","pitch_jitter_cents"),dash:[6,2],color:COLORS[3]}]));
  C.append(chart("Validation score (lower is better)",[{name:"score",points:flat(ev,"val",e=>e.morph?e.morph.score:null),dots:true,color:COLORS[0]}]));
  C.append(chart("Learning rate",[{name:"lr",points:series(ev,"train",e=>e.lr)}],{log:true}));
  C.append(chart("Seconds per step",[{name:"s/step",points:smooth(flat(ev,"train",e=>e.sec_per_step),8),color:COLORS[8]}]));
  renderMorphPanel(ev)}
function renderDiffusion(C,ev){
  const L=k=>flat(ev,"train",e=>e.losses?e.losses[k]:null),Q=(set,k)=>flat(ev,"val",e=>e.diffusion&&e.diffusion[set]?e.diffusion[set][k]:null);
  const loss=L("denoise");
  C.append(chart("Denoising loss (v-prediction)",[{name:"per log step",points:loss,color:"#94a3b8",width:1},{name:"smoothed",points:smooth(loss,8),color:COLORS[0],width:2}]));
  const vv=Q("vox","vocality_A");
  C.append(chart("Quick check: vocality of generated notes (judge A; vox → 1, synth → 0, 'and' α 0.5 → 0.5)",[
    {name:"vox samples",points:vv,dots:true,color:COLORS[0]},{name:"synth samples",points:Q("synth","vocality_A"),dots:true,color:COLORS[1]},
    {name:"'and' α 0.5",points:Q("and_0.5","vocality_A"),dots:true,color:COLORS[2]},refLine("0.5",vv,0.5)]));
  C.append(chart("Quick check: 'and' α 0.5 samples with both detectors on",[{name:"judge A",points:Q("and_0.5","both_present_A"),dots:true,color:COLORS[0]},
    {name:"judge B",points:Q("and_0.5","both_present_B"),dots:true,color:COLORS[1]},refLine("required 0.5",Q("and_0.5","both_present_A"),0.5,"#b91c1c")]));
  C.append(chart("Quick check: KID in judge-A embeddings vs real held-out notes (lower = more real)",[{name:"vox samples",points:Q("vox","kid_A"),dots:true,color:COLORS[0]},
    {name:"synth samples",points:Q("synth","kid_A"),dots:true,color:COLORS[1]},{name:"'and' α 0.5 (vs all classes)",points:Q("and_0.5","kid_A"),dots:true,color:COLORS[2]}]));
  C.append(chart("Checkpoint score (fidelity + realism + both-on; lower is better)",[{name:"score",points:flat(ev,"val",e=>e.score),dots:true,color:COLORS[0]}]));
  C.append(chart("Learning rate",[{name:"lr",points:series(ev,"train",e=>e.lr)}],{log:true}));
  C.append(chart("Seconds per step",[{name:"s/step",points:smooth(flat(ev,"train",e=>e.sec_per_step),8),color:COLORS[8]}]));
  renderMorphPanel(ev)}
function renderMorphPanel(ev){
  const M=$("morph"),judges=ev.filter(e=>e.type==="judge"),demos=ev.filter(e=>e.type==="demos"),sig=state.run+":"+judges.length+":"+demos.length;
  if(M.dataset.sig===sig)return;M.dataset.sig=sig;let html="";
  if(judges.length){html+="<h2>Judges (trained before the generator; J is never used for generation)</h2>";
    judges.forEach(j=>{const rows={};Object.entries(j.report).forEach(([set,v])=>rows[set]=v);
      html+=`<h2 class=muted>${j.name}</h2>`+table(rows,[...new Set(Object.values(rows).flatMap(Object.keys))])})}
  if(demos.length){const keep=$("demoSel")?$("demoSel").value:null;
    html+=`<h2>Listening <select id=demoSel>${demos.map((d,i)=>`<option value=${i}>epoch ${d.epoch}</option>`).join("")}</select>
      <span class=muted style="font-weight:400"> spectrograms <select id=demoSpec><option value=sharp>sharp</option><option value=exact>exact samples</option><option value=off>off</option></select>
      (mid, 27 Hz – 21 kHz, each re its own loudest point; hover for frequency and time)</span></h2><div id=demoBody></div>`;
    M.innerHTML=html;const sel=$("demoSel"),spec=$("demoSpec");sel.value=keep!=null&&demos[keep]?keep:String(demos.length-1);
    const ORDER=["hybrid","hybrid, vocal pitch","vocal","synth","mix","vocoder"];
    const draw=()=>{const d=demos[sel.value],seen=[...new Set(d.pairs.flatMap(p=>Object.keys(p.files)))],cols=[...ORDER.filter(k=>seen.includes(k)),...seen.filter(k=>!ORDER.includes(k))],url=f=>encodeURIComponent(d.folder+"/"+f);
      $("demoBody").innerHTML=`<div class=scroll style="max-height:none"><table><tr><th>pair</th>${cols.map(k=>`<th style="text-align:left">${k==="hybrid"?"hybrid (synth pitch)":k}</th>`).join("")}</tr>`+
      d.pairs.map(p=>`<tr><td style="white-space:normal;max-width:240px;vertical-align:top"><b>${noteName(p.midi)}</b> · α ${p.alpha}<br><span class=muted>${p.vocal}<br>${p.synth}</span></td>`+
        cols.map(k=>p.files[k]?`<td style="text-align:left;vertical-align:top"><audio controls preload=none style="width:190px" src="/api/file?name=${encodeURIComponent(state.run)}&path=${url(p.files[k])}"></audio>`+
          (spec.value!=="off"?`<canvas class=demospec data-path="${d.folder+"/"+p.files[k]}"></canvas>`:"")+"</td>":"<td></td>").join("")+"</tr>").join("")+"</table></div>";
      requestAnimationFrame(()=>document.querySelectorAll("canvas.demospec").forEach(c=>demoSpectrogram(c,c.dataset.path,spec.value)))};
    sel.onchange=draw;spec.onchange=draw;draw()}
  else{html+="<p class=muted>Listening demos appear after the first full epoch.</p>";M.innerHTML=html}
}
function demoSpectrogram(canvas,path,mode){const {g,dpr,w,h}=frame(canvas),W=Math.round(w*dpr),H=Math.round(h*dpr),m0=21,m1=136;
  g.fillStyle=css("--muted");g.font="10px system-ui";g.fillText("…",4,12);
  fetchBytesRun("/api/demo/view",{path,mode,w:W,h:H,m0,m1}).then(b=>{const img=paint(b,W,H,"level");g.clearRect(0,0,w,h);g.imageSmoothingEnabled=false;g.drawImage(img,0,0,w,h);
    g.font="9px system-ui";[[100,"100"],[1000,"1k"],[10000,"10k"]].forEach(([hz,lab])=>{const y=h-(hzMidi(hz)-m0)/(m1-m0)*h,tw=g.measureText(lab).width+4;
      g.fillStyle=css("--panel");g.globalAlpha=0.7;g.fillRect(1,y-6,tw,10);g.globalAlpha=1;g.fillStyle=css("--muted");g.fillText(lab,3,y+3)});
    canvas.onmousemove=e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,mi=m0+(1-y/h)*(m1-m0),px=Math.min(W-1,Math.floor(x/w*W)),py=Math.min(H-1,Math.floor((1-y/h)*H)),lv=b[py*W+px];
      tip(e,`<b>${fmtHz(midiHz(mi))}</b> (${noteName(Math.round(mi))}) · ${(x/w*1200).toFixed(0)} ms<br>${lv===255?"below −90 dB":(-90+lv/254*90).toFixed(1)+" dB re loudest"}`)};
    canvas.onmouseleave=e=>tip(e,null)}).catch(err=>{g.clearRect(0,0,w,h);g.fillText("spectrogram unavailable",4,12)})}
async function fetchBytesRun(path,p){const qs=new URLSearchParams({name:state.run,...p}).toString(),key=path+"?"+qs;if(vcache.has(key))return vcache.get(key);
  const r=await fetch(key);if(!r.ok)throw new Error(await r.text());const b=new Uint8Array(await r.arrayBuffer());vcache.set(key,b);if(vcache.size>200)vcache.delete(vcache.keys().next().value);return b}
function renderProbes(){
  const probes=state.data.events.filter(e=>e.type==="probe");const sel=$("probeFeature");
  const feats=probes.length?Object.keys(probes[0].results):[];const keep=sel.value;
  sel.innerHTML=feats.map(f=>`<option>${f}</option>`).join("");if(feats.includes(keep))sel.value=keep;else if(feats.includes("stage4"))sel.value="stage4";
  const P=$("probeCharts");P.innerHTML="";if(!probes.length){P.innerHTML="<p class=muted>Probe events appear at each checkpoint of runs started with the new training script.</p>";return}
  const f=sel.value,targets=Object.keys(probes[0].results[f]);
  const reg=targets.filter(t=>"r2" in probes[0].results[f][t]),cls=targets.filter(t=>!("r2" in probes[0].results[f][t]));
  P.append(chart(`R² of regression probes (${f})`,reg.map(t=>({name:t,points:probes.map(e=>[e.epoch,e.results[f][t].r2]),dots:true}))));
  P.append(chart(`Accuracy of classification probes (${f}); dashed = majority class`,cls.flatMap((t,i)=>[{name:t,points:probes.map(e=>[e.epoch,e.results[f][t].accuracy]),dots:true,color:COLORS[i]},
    {name:t+" majority",points:probes.map(e=>[e.epoch,e.results[f][t].majority_baseline]),dash:[3,3],color:COLORS[i]}])));
}
function renderCkpts(){
  const rows=state.data.checkpoints,task=(state.data.info.run||{}).task,valHead=task==="morph"?"val score (lower is better)":task==="diffusion"?"check score (lower is better)":"val error (dB)";
  $("ckpts").innerHTML=`<tr><th>file</th><th>epoch</th><th>${valHead}</th><th>size</th><th>saved</th></tr>`+
    rows.map(r=>`<tr class="${r.best?"best":""}"><td>${r.file}${r.best?" ★ best":""}</td><td>${r.epoch??""}</td><td>${r.val_model??""}</td><td>${r.size_mb} MB</td><td>${r.modified}</td></tr>`).join("");
}
function table(obj,cols){const keys=Object.keys(obj);if(!keys.length)return"";cols=cols||Object.keys(obj[keys[0]]);
  return`<div class=scroll><table><tr><th></th>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>${keys.map(k=>`<tr><td>${k}</td>${cols.map(c=>`<td>${fmt(obj[k][c])}</td>`).join("")}</tr>`).join("")}</table></div>`}
function fmt(v){return v==null?"":typeof v==="object"?JSON.stringify(v):v}
function renderEvals(){
  const evs=state.data.evaluations,sel=$("evalSel"),keep=sel.value;sel.innerHTML=evs.map((e,i)=>`<option value=${i}>${e.file} (${e.modified})</option>`).join("");
  if(keep&&evs[keep])sel.value=keep;else sel.value=evs.length-1;const e=evs[sel.value];const B=$("evalBody");
  if(!e){B.innerHTML="<p class=muted>No evaluation reports yet.</p>";return}const r=e.report;let html="";
  const acc=r.accuracy||r.test;if(acc){const flat={};Object.entries(acc).forEach(([k,v])=>{if(k!=="per_family")flat[k]=v});html+="<h2>Accuracy (envelope error, dB)</h2>"+table(flat);
    if(acc.per_family)html+="<h2>Per sound family</h2>"+table(acc.per_family)}
  if(r.head){html+="<h2>Head: correction delivered</h2>"+table({correction:r.head.correction});html+="<h2>Head: ablations (dB)</h2>"+table(r.head.ablations);
    html+="<h2>Head: auxiliary predictions (mean abs error)</h2>"+table(r.head.aux);html+="<h2>Head: error by region (dB)</h2>"+table(r.head.error_by_region)}
  if(r.backbone){html+="<h2>Backbone: per stage</h2>"+table(r.backbone.per_stage)+"<h2>Backbone: feature effective rank</h2>"+table(r.backbone.feature_rank)}
  if(r.verdict){html+=`<h2>Verdict: ${r.verdict.decision} <span class=muted>(best 'and' setting: ${r.verdict.best_and_setting})</span></h2>`+
    table(Object.fromEntries(Object.entries(r.verdict.passes).map(([k,v])=>[k,{pass:v?"yes":"NO",criterion:(r.criteria||{})[k]||""}])),["pass","criterion"]);
    if(r.classes)html+="<h2>Generated notes per class</h2>"+table(r.classes,[...new Set(Object.values(r.classes).flatMap(Object.keys))]);
    if(r.references)html+="<h2>References on held-out pairs (same judges, same KID space)</h2>"+table(r.references,[...new Set(Object.values(r.references).flatMap(Object.keys))]);
    if(r.composition)html+="<h2>Composition grid</h2>"+table(r.composition,[...new Set(Object.values(r.composition).flatMap(Object.keys))]);
    if(r.automation)html+="<h2>α automated 0 → 1 within the note (vocality, judge A)</h2>"+table({vocality:r.automation})}
  if(r.probes){Object.entries(r.probes).forEach(([ck,res])=>{const rows={};Object.entries(res).forEach(([feat,t])=>{rows[feat]={};Object.entries(t).forEach(([tn,m])=>rows[feat][tn]=m.r2??m.accuracy)});
    html+=`<h2>Probes: ${ck} (R² or accuracy, held-out families)</h2>`+table(rows)})}
  B.innerHTML=html||"<pre>"+JSON.stringify(r,null,1).slice(0,5000)+"</pre>";
}
// ---------- encoding examples ----------
let enc={run:null,data:null};
const dark=()=>window.matchMedia("(prefers-color-scheme: dark)").matches;
function hex(h){h=h.replace("#","");return[0,2,4].map(i=>parseInt(h.slice(i,i+2),16))}
function ramp(stops,t){t=Math.max(0,Math.min(1,t));const n=stops.length-1,i=Math.min(n-1,Math.floor(t*n)),f=t*n-i,a=hex(stops[i]),b=hex(stops[i+1]);return a.map((v,k)=>Math.round(v+(b[k]-v)*f))}
function palette(kind){if(kind==="div")return dark()?["#86b6ef","#383835","#e66767"]:["#184f95","#f0efec","#e34948"];
  return dark()?["#1a1a19","#184f95","#3987e5","#86b6ef","#cde2fb"]:["#fcfcfb","#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d366b"]}
function decode64(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
function noteName(m){return"C C# D D# E F F# G G# A A# B".split(" ")[m%12]+(Math.floor(m/12)-1)}
function fmtHz(f){return f>=1000?(f/1000).toFixed(f>=10000?1:2)+" kHz":f.toFixed(1)+" Hz"}
function tip(e,html){const t=$("tip");if(!html){t.style.display="none";return}t.innerHTML=html;t.style.display="block";t.style.left=(e.clientX+14)+"px";t.style.top=(e.clientY+12)+"px"}
function heatmap(canvas,plane,info,example){
  const rows=info.rows,frames=info.frames,data=plane._bytes||(plane._bytes=decode64(plane.data));
  const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;
  const g=canvas.getContext("2d");g.scale(dpr,dpr);const L=46,R=4,T=4,B=18,pw=w-L-R,ph=h-T-B;
  const img=new ImageData(frames,rows),stops=palette(plane.kind),surf=hex(css("--panel"));
  for(let r=0;r<rows;r++)for(let c=0;c<frames;c++){const v=data[r*frames+c],o=((rows-1-r)*frames+c)*4;const col=v===255?surf:ramp(stops,v/254);img.data[o]=col[0];img.data[o+1]=col[1];img.data[o+2]=col[2];img.data[o+3]=255}
  const off=document.createElement("canvas");off.width=frames;off.height=rows;off.getContext("2d").putImageData(img,0,0);
  g.imageSmoothingEnabled=false;g.drawImage(off,L,T,pw,ph);
  const ink=css("--muted");g.fillStyle=ink;g.strokeStyle=css("--line");g.font="10px system-ui";
  const rowOf=hz=>12*Math.log2(hz/440)+69-info.row_midi[0];
  [[20,"20 Hz"],[100,"100 Hz"],[1000,"1 kHz"],[10000,"10 kHz"],[24000,"24 kHz"],[80000,"80 kHz"]].forEach(([hz,lab])=>{const r=rowOf(hz);if(r<0||r>rows-1)return;const y=T+ph-(r+0.5)/rows*ph;g.fillText(lab,2,y+3);g.beginPath();g.moveTo(L-3,y);g.lineTo(L,y);g.stroke()});
  [0,300,600,900,1200].forEach(ms=>{const x=L+ms/(frames*info.hop_ms)*pw,lab=ms+(ms===1200?" ms":"");g.textAlign=ms===1200?"right":ms===0?"left":"center";g.fillText(lab,ms===1200?w-R:x,h-5)});g.textAlign="left";
  const top=example.nyquist_rows.lastIndexOf(1);
  if(example.source_rate<example.stored_rate&&top<rows-1){const y=T+ph-(top+1)/rows*ph;g.setLineDash([4,3]);g.strokeStyle=ink;g.beginPath();g.moveTo(L,y);g.lineTo(L+pw,y);g.stroke();g.setLineDash([]);
    g.fillText("source Nyquist "+fmtHz(example.source_rate/2)+": rows above not scored",L+4,y-3)}
  canvas.onmousemove=e=>{const b=canvas.getBoundingClientRect(),x=e.clientX-b.left-L,y=e.clientY-b.top-T;
    if(x<0||y<0||x>pw||y>ph)return tip(e,null);const c=Math.floor(x/pw*frames),r=rows-1-Math.floor(y/ph*rows),v=data[r*frames+c];
    const val=v===255?"quiet cell (below −90 dB)":(plane.min+(v/254)*(plane.max-plane.min)).toFixed(plane.unit==="dB"?1:3)+" "+plane.unit;
    tip(e,`<b>${noteName(info.row_midi[r])}</b> row (${fmtHz(info.row_hz[r])}) · ${(c*info.hop_ms).toFixed(0)}–${((c+1)*info.hop_ms).toFixed(0)} ms<br>${val}<br><span class=muted>${info.row_samples_per_cell[r]} complex sample${info.row_samples_per_cell[r]>1?"s":""} per channel in this cell${example.nyquist_rows[r]?"":" · above source Nyquist"}</span>`)};
  canvas.onmouseleave=e=>tip(e,null);
}
function scaleBar(plane){const c=document.createElement("canvas");c.width=160;c.height=10;const g=c.getContext("2d"),stops=palette(plane.kind);
  for(let x=0;x<160;x++){const col=ramp(stops,x/159);g.fillStyle=`rgb(${col})`;g.fillRect(x,0,1,10)}
  const d=document.createElement("div");d.className="scale";d.append(String(plane.min),c,String(plane.max)+(plane.unit?" "+plane.unit:""));return d}
function waveform(canvas,ex){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;
  const g=canvas.getContext("2d");g.scale(dpr,dpr);const cols=dark()?["#3987e5","#d95926"]:["#2a78d6","#eb6834"];
  const peak=Math.max(1e-9,...["left","right"].flatMap(k=>ex.waveform[k].max.concat(ex.waveform[k].min.map(Math.abs))));
  ["left","right"].forEach((k,i)=>{const mn=ex.waveform[k].min,mx=ex.waveform[k].max,n=mn.length,y0=(i+0.5)*h/2,a=h/4.4;g.fillStyle=cols[i];
    for(let j=0;j<n;j++){const x=j/n*w,top=y0-mx[j]/peak*a,bot=y0-mn[j]/peak*a;g.fillRect(x,top,Math.max(1,w/n),Math.max(1,bot-top))}
    g.font="10px system-ui";const tw=g.measureText(k).width+6;g.fillStyle=css("--panel");g.fillRect(0,i*h/2+1,tw,12);g.fillStyle=css("--muted");g.fillText(k,3,i*h/2+10)})}
function cellLayout(canvas,info){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;
  const g=canvas.getContext("2d");g.scale(dpr,dpr);const n=info.rows,L=46,B=16,pw=w-L-4,ph=h-B-4,col=dark()?"#3987e5":"#2a78d6";
  info.row_samples_per_cell.forEach((s,r)=>{const bh=(Math.log2(s)+1)/9*ph;g.fillStyle=col;g.fillRect(L+r/n*pw+0.5,4+ph-bh,Math.max(1,pw/n-1),bh)});
  g.fillStyle=css("--muted");g.font="10px system-ui";[1,4,16,64,256].forEach(s=>{const y=4+ph-(Math.log2(s)+1)/9*ph;g.fillText(String(s),2,y+3)});
  [[20,"20 Hz"],[100,"100 Hz"],[1000,"1 kHz"],[10000,"10 kHz"],[80000,"80 kHz"]].forEach(([hz,lab])=>{const r=12*Math.log2(hz/440)+69-info.row_midi[0];g.textAlign=hz===80000?"right":"center";g.fillText(lab,Math.min(L+r/n*pw,w-2),h-3)});g.textAlign="left";
  canvas.onmousemove=e=>{const b=canvas.getBoundingClientRect(),r=Math.floor((e.clientX-b.left-L)/pw*n);if(r<0||r>=n)return tip(e,null);
    tip(e,`<b>${noteName(info.row_midi[r])}</b> row (${fmtHz(info.row_hz[r])})<br>${info.row_samples_per_cell[r]} exact complex sample(s) per 10 ms per channel`)};canvas.onmouseleave=e=>tip(e,null)}
const vcache=new Map();
function frame(canvas){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const g=canvas.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);return{g,dpr,w,h}}
async function fetchBytes(path,p){const qs=new URLSearchParams({name:enc.run,...p}).toString(),key=path+"?"+qs;if(vcache.has(key))return vcache.get(key);
  const r=await fetch(key);if(!r.ok)throw new Error((await r.text()).replace(/<[^>]+>/g," ").trim().slice(0,200));
  const b=new Uint8Array(await r.arrayBuffer());vcache.set(key,b);if(vcache.size>80)vcache.delete(vcache.keys().next().value);return b}
const hzMidi=hz=>69+12*Math.log2(hz/440),midiHz=m=>440*Math.pow(2,(m-69)/12);
function seqLut(){const st=palette("seq");return Array.from({length:255},(_,v)=>ramp(st,v/254))}
function stereoStops(mode){if(mode==="pan")return dark()?["#6ea8f0","#e8e8e3","#f0915c"]:["#1c5cab","#3a3a37","#c2410c"];
  return dark()?["#f07070","#6ea8f0","#e8e8e3"]:["#c62828","#1c5cab","#3a3a37"]}
function paint(bytes,W,H,mode){const img=new ImageData(W,H),d=img.data,surf=hex(css("--panel")),n=W*H,stereo=mode==="pan"||mode==="corr";
  const lut=stereo?Array.from({length:255},(_,v)=>ramp(stereoStops(mode),v/254)):seqLut();
  for(let y=0;y<H;y++){const src=(H-1-y)*W;for(let x=0;x<W;x++){const i=src+x,o=(y*W+x)*4,lv=bytes[i];let c;
    if(lv===255)c=surf;else if(stereo){const hue=lut[bytes[n+i]],a=Math.pow(lv/254,0.7);c=[surf[0]+(hue[0]-surf[0])*a,surf[1]+(hue[1]-surf[1])*a,surf[2]+(hue[2]-surf[2])*a]}else c=lut[lv];
    d[o]=c[0];d[o+1]=c[1];d[o+2]=c[2];d[o+3]=255}}
  const off=document.createElement("canvas");off.width=W;off.height=H;off.getContext("2d").putImageData(img,0,0);return off}
function freqTicks(m0,m1,px){const span=m1-m0,out=[];
  if(span<=30){for(let m=Math.ceil(m0);m<=m1;m++)if(span<=14||m%12===0)out.push([m,noteName(m)+" · "+fmtHz(midiHz(m))])}
  else[20,50,100,200,500,1000,2000,5000,10000,20000,50000].forEach(hz=>{const m=hzMidi(hz);if(m>=m0&&m<=m1)out.push([m,hz>=1000?hz/1000+" kHz":hz+" Hz"])});
  let last=-1e9;return out.filter(([m])=>{const y=(m-m0)/span*px;if(y-last<13)return false;last=y;return true})}
function timeTicks(t0,t1,px){const span=(t1-t0)*1000,steps=[0.5,1,2,5,10,20,50,100,200,500];const step=steps.find(s=>span/s<=Math.max(3,px/70))||1000;
  const out=[];for(let ms=Math.ceil(t0*1000/step)*step;ms<=t1*1000+1e-9;ms+=step)out.push(ms);return out}
const fullView=()=>({t0:0,t1:1.2,m0:11.5,m1:160.5});
function sharpMap(canvas,cfg){
  const {g,dpr,w,h}=frame(canvas),big=!!cfg.zoom,L=big?92:46,R=6,T=6,B=20,pw=w-L-R,ph=h-T-B,v=cfg.view||fullView();
  const W=Math.max(16,Math.round(pw*dpr)),H=Math.max(16,Math.round(ph*dpr)),token=(canvas._token=(canvas._token||0)+1);
  const ex=cfg.example,info=cfg.info;let base=null,bytes=null,drag=null;
  function axes(){g.fillStyle=css("--muted");g.strokeStyle=css("--line");g.font=(big?11:10)+"px system-ui";
    freqTicks(v.m0,v.m1,ph).forEach(([m,lab])=>{const y=T+ph-(m-v.m0)/(v.m1-v.m0)*ph;g.fillText(lab,2,y+3);g.beginPath();g.moveTo(L-3,y);g.lineTo(L,y);g.stroke()});
    const tt=timeTicks(v.t0,v.t1,pw);tt.forEach((ms,k)=>{const x=L+(ms/1000-v.t0)/(v.t1-v.t0)*pw,lab=(ms%1?ms.toFixed(1):ms)+(k===tt.length-1?" ms":"");
      g.textAlign=x>w-30?"right":x<L+20?"left":"center";g.fillText(lab,Math.min(x,w-R),h-5)});g.textAlign="left";
    const nyq=hzMidi(ex.source_rate/2);if(ex.source_rate<ex.stored_rate&&nyq>v.m0&&nyq<v.m1){const y=T+ph-(nyq-v.m0)/(v.m1-v.m0)*ph;
      g.setLineDash([4,3]);g.strokeStyle=css("--muted");g.beginPath();g.moveTo(L,y);g.lineTo(L+pw,y);g.stroke();g.setLineDash([]);
      const lab="source Nyquist "+fmtHz(ex.source_rate/2)+": not scored above",tw=g.measureText(lab).width+6;g.fillStyle=css("--panel");g.fillRect(L+3,y-14,tw,12);g.fillStyle=css("--muted");g.fillText(lab,L+6,y-4)}}
  function paintAll(){g.clearRect(0,0,w,h);if(base){g.imageSmoothingEnabled=false;g.drawImage(base,L,T,pw,ph)}else{g.fillStyle=css("--muted");g.font="12px system-ui";g.textAlign="center";g.fillText(cfg.fallback||"computing…",L+pw/2,T+ph/2);g.textAlign="left"}
    axes();if(drag){g.strokeStyle=css("--ink");g.setLineDash([3,2]);g.strokeRect(Math.min(drag.x0,drag.x1),Math.min(drag.y0,drag.y1),Math.abs(drag.x1-drag.x0),Math.abs(drag.y1-drag.y0));g.setLineDash([])}}
  if(cfg.placeholder){base=cfg.placeholder}
  paintAll();if(cfg.status)cfg.status("computing…");
  const t=performance.now();
  fetchBytes("/api/encoding/view",{ex:cfg.ex,src:cfg.src||"note",mode:cfg.mode,show:cfg.show||"level",ch:cfg.ch||"mid",t0:v.t0,t1:v.t1,m0:v.m0,m1:v.m1,w:W,h:H}).then(b=>{
    if(token!==canvas._token)return;bytes=b;base=paint(b,W,H,cfg.show||"level");
    if(!b.subarray(0,W*H).some(x=>x!==255))cfg.fallback=null,base=null,cfg.fallback=cfg.ch==="side"&&cfg.example.side_to_mid_db<-200?"side is exactly 0: mono source":"silent in this range";paintAll();if(cfg.status)cfg.status(`${W} × ${H} px, ${Math.round(performance.now()-t)} ms`)}).catch(err=>{
    if(token!==canvas._token)return;if(cfg.status)cfg.status("sharp view unavailable: "+err.message);cfg.fallback="sharp view unavailable";paintAll()});
  const at=e=>{const b=canvas.getBoundingClientRect();return[e.clientX-b.left,e.clientY-b.top]};
  canvas.onmousemove=e=>{const[x,y]=at(e);if(drag){drag.x1=Math.max(L,Math.min(L+pw,x));drag.y1=Math.max(T,Math.min(T+ph,y));paintAll();return}
    if(x<L||y<T||x>L+pw||y>T+ph)return tip(e,null);const tm=v.t0+(x-L)/pw*(v.t1-v.t0),m=v.m0+(1-(y-T)/ph)*(v.m1-v.m0),hz=midiHz(m),near=Math.round(m),cents=Math.round((m-near)*100);
    let val="";if(bytes){const px=Math.min(W-1,Math.floor((x-L)/pw*W)),py=Math.min(H-1,Math.floor((1-(y-T)/ph)*H)),i=py*W+px,lv=bytes[i];
      val=lv===255?"below −100 dB":`${(-100+lv/254*100).toFixed(1)} dB re loudest`;
      if(cfg.show==="pan"&&lv!==255){const p=bytes[W*H+i]/127-1;val+=` · balance ${p>0.02?"R ":p<-0.02?"L ":""}${Math.abs(p).toFixed(2)}`}
      if(cfg.show==="corr"&&lv!==255)val+=` · correlation ${(bytes[W*H+i]/127-1).toFixed(2)}`}
    tip(e,`<b>${fmtHz(hz)}</b> (${noteName(near)} ${cents>=0?"+":""}${cents} ct) · ${(tm*1000).toFixed(tm<0.1?2:1)} ms<br>${val}${big?"<br><span class=muted>drag to zoom · double-click to zoom out</span>":""}`)};
  canvas.onmouseleave=e=>{tip(e,null)};
  if(!big){canvas.onmousedown=null;canvas.ondblclick=null;return}
  canvas.onmousedown=e=>{const[x,y]=at(e);if(x<L||y<T||x>L+pw||y>T+ph)return;drag={x0:x,y0:y,x1:x,y1:y};e.preventDefault()};
  window.onmouseup=e=>{if(!drag)return;const d=drag;drag=null;const dx=Math.abs(d.x1-d.x0),dy=Math.abs(d.y1-d.y0);if(dx<6&&dy<6){paintAll();return}
    const tx=x=>v.t0+(x-L)/pw*(v.t1-v.t0),my=y=>v.m0+(1-(y-T)/ph)*(v.m1-v.m0);let nv={...v};
    if(dx>=6){nv.t0=tx(Math.min(d.x0,d.x1));nv.t1=tx(Math.max(d.x0,d.x1));if(nv.t1-nv.t0<0.01){const c=(nv.t0+nv.t1)/2;nv.t0=c-0.005;nv.t1=c+0.005}}
    if(dy>=6){nv.m0=my(Math.max(d.y0,d.y1));nv.m1=my(Math.min(d.y0,d.y1));if(nv.m1-nv.m0<1){const c=(nv.m0+nv.m1)/2;nv.m0=c-0.5;nv.m1=c+0.5}}
    cfg.onzoom(nv)};
  canvas.ondblclick=e=>{const[x,y]=at(e),f=fullView(),tc=v.t0+(x-L)/pw*(v.t1-v.t0),mc=v.m0+(1-(y-T)/ph)*(v.m1-v.m0);
    const span=(a,b,c,lo,hi)=>{let s=Math.min(hi-lo,(b-a)*2),s0=Math.max(lo,Math.min(hi-s,c-s/2));return[s0,s0+s]};
    const[t0,t1]=span(v.t0,v.t1,tc,f.t0,f.t1),[m0,m1]=span(v.m0,v.m1,mc,f.m0,f.m1);cfg.onzoom({t0,t1,m0,m1})};
}
function placeholderFrom(plane,info){if(!plane)return null;const data=plane._bytes||(plane._bytes=decode64(plane.data)),off=document.createElement("canvas");
  off.width=info.frames;off.height=info.rows;const img=new ImageData(info.frames,info.rows),lut=seqLut(),surf=hex(css("--panel"));
  for(let r=0;r<info.rows;r++)for(let c=0;c<info.frames;c++){const v=data[r*info.frames+c],o=((info.rows-1-r)*info.frames+c)*4,col=v===255?surf:lut[v];img.data[o]=col[0];img.data[o+1]=col[1];img.data[o+2]=col[2];img.data[o+3]=255}
  off.getContext("2d").putImageData(img,0,0);return off}
function stereoScale(mode){const c=document.createElement("canvas");c.width=160;c.height=10;const g=c.getContext("2d"),st=stereoStops(mode);
  for(let x=0;x<160;x++){const col=ramp(st,x/159);g.fillStyle=`rgb(${col})`;g.fillRect(x,0,1,10)}
  const d=document.createElement("div");d.className="scale";const ends=mode==="pan"?["left","right"]:["opposite phase","in phase (mono)"];
  d.append(ends[0],c,ends[1]+(mode==="corr"?" · middle = unrelated (wide)":"")+" · brightness = level, −100 … 0 dB");return d}
function levelScale(){const d=scaleBar({kind:"seq",min:-100,max:0,unit:"dB"});d.append(" re the loudest pixel of the whole note");return d}
async function gonio(canvas,e,i){const {g,dpr,w,h}=frame(canvas),side=Math.min(w,h),x0=(w-side)/2,size=Math.round(side*dpr);
  const lineCol=css("--line"),ink=css("--muted");
  function guides(){g.strokeStyle=lineCol;g.lineWidth=1;g.setLineDash([3,3]);const c=x0+side/2,r=side/2;
    [[0,-1],[1,0],[-0.7071,-0.7071],[0.7071,-0.7071]].forEach(([dx,dy])=>{g.beginPath();g.moveTo(c-dx*r,h/2-dy*r);g.lineTo(c+dx*r,h/2+dy*r);g.stroke()});g.setLineDash([]);
    g.fillStyle=ink;g.font="11px system-ui";g.fillText("M",c+4,14);g.fillText("L",x0+12,26);g.textAlign="right";g.fillText("R",x0+side-12,26);g.fillText("S",x0+side-2,h/2-4);g.textAlign="left"}
  guides();
  try{const b=await fetchBytes("/api/encoding/scope",{ex:i,size});const img=new ImageData(size,size),lut=seqLut(),surf=hex(css("--panel"));
    for(let k=0;k<size*size;k++){const c=b[k]===255?surf:lut[b[k]];img.data[k*4]=c[0];img.data[k*4+1]=c[1];img.data[k*4+2]=c[2];img.data[k*4+3]=255}
    const off=document.createElement("canvas");off.width=size;off.height=size;off.getContext("2d").putImageData(img,0,0);
    g.clearRect(0,0,w,h);g.imageSmoothingEnabled=false;g.drawImage(off,x0,(h-side)/2,side,side);guides()}
  catch(err){g.fillStyle=ink;g.fillText("goniometer unavailable",x0+8,h/2)}}
function corrMeter(e){const s=e.stereo,pct=v=>((v+1)/2*100).toFixed(1)+"%",bal=s.balance;
  return`<div class=meter><span>−1</span><div class=bar style="background:linear-gradient(90deg,${stereoStops("corr").join(",")})"><i style="left:${pct(s.correlation)}"></i></div><span>+1</span></div>
  <div class=muted style="font-size:12px">phase correlation <b>${s.correlation>=0?"+":""}${s.correlation.toFixed(2)}</b> · balance <b>${Math.abs(bal)<0.02?"centre":(bal<0?"L ":"R ")+Math.abs(bal).toFixed(2)}</b> · side/mid ${e.side_to_mid_db<-200?"−∞ (mono)":e.side_to_mid_db+" dB"}</div>`}
function stereoByFreq(canvas,e,info){const {g,w,h}=frame(canvas),L=30,R=6,T=8,B=18,pw=w-L-R,ph=h-T-B,n=info.rows,s=e.stereo.per_row;
  const X=r=>L+(r+0.5)/n*pw,Y=v=>T+(1-(v+1)/2)*ph;g.font="10px system-ui";g.fillStyle=css("--muted");g.strokeStyle=css("--line");
  [-1,0,1].forEach(v=>{g.beginPath();g.moveTo(L,Y(v));g.lineTo(L+pw,Y(v));g.stroke();g.fillText((v>0?"+":"")+v,4,Y(v)+3)});
  [[100,"100 Hz"],[1000,"1 kHz"],[10000,"10 kHz"],[50000,"50 kHz"]].forEach(([hz,lab])=>{const r=hzMidi(hz)-info.row_midi[0];g.textAlign="center";g.fillText(lab,X(r),h-4)});g.textAlign="left";
  const live=r=>s.level_db[r]>-60;const ser=[["correlation",COLORS[2]],["balance",COLORS[4]]];
  ser.forEach(([k,col])=>{g.strokeStyle=col;g.lineWidth=2;g.beginPath();let on=false;for(let r=0;r<n;r++){if(!live(r)){on=false;continue}const x=X(r),y=Y(s[k][r]);on?g.lineTo(x,y):g.moveTo(x,y);on=true}g.stroke()});g.lineWidth=1;
  canvas.onmousemove=ev=>{const b=canvas.getBoundingClientRect(),r=Math.floor((ev.clientX-b.left-L)/pw*n);if(r<0||r>=n)return tip(ev,null);
    tip(ev,`<b>${noteName(info.row_midi[r])}</b> row (${fmtHz(info.row_hz[r])})<br>`+(live(r)?`correlation ${s.correlation[r].toFixed(2)} · balance ${s.balance[r].toFixed(2)}<br><span class=muted>level ${s.level_db[r]} dB re loudest row</span>`:`<span class=muted>below −60 dB: not drawn</span>`))};canvas.onmouseleave=ev=>tip(ev,null)}
const dv={ex:0,show:"level",ch:"mid",src:"note",view:null},pic={mode:"sharp"};
const PICTURES=[["sharp","Sharp (reassigned)"],["exact","Exact samples"],["cells","Cell grid (backbone)"]];
const PIC_ABOUT={sharp:"Sharp: each sample's energy moved to its exact instantaneous frequency (phase derivative) and its own time (group delay), as far as neighbouring bands agree on the move; noise stays smooth. Computed only from the stored image.",
  exact:"Exact: each band's level at its own time resolution (band-limited interpolation of its samples), blended between band centres. No reassignment: the time–frequency trade-off of the bands themselves.",
  cells:"Cell grid: one value per 10 ms × semitone cell, the grid the backbone's feature planes are computed on (its patches then group 4 × 4 cells)."};
const SHOW_ABOUT={level:"",pan:" Colour = left/right balance (L = mid + side, R = mid − side), brightness = level.",
  corr:" Colour = phase correlation of left and right: in phase (mono-compatible), unrelated (wide) or opposite (cancels in mono); brightness = level."};
function seg(id,items,cur,on){const el=$(id);el.innerHTML=items.map(([k,lab])=>`<button data-k="${k}" class="${k===cur?"on":""}">${lab}</button>`).join("");
  el.onclick=e=>{const k=e.target.dataset&&e.target.dataset.k;if(k!=null)on(k)}}
function renderViewer(){const d=enc.data,ex=d.examples,stereo=dv.show!=="level";
  seg("dvEx",ex.map((e,i)=>[String(i),`${i+1}. ${e.label.split(",")[0]} · ${(e.source_rate/1000).toFixed(1)} kHz`]),String(dv.ex),k=>{dv.ex=+k;dv.view=null;renderViewer()});
  seg("dvShow",[["level","Level"],["pan","Stereo pan"],["corr","Phase correlation"]],dv.show,k=>{dv.show=k;renderViewer()});
  seg("dvCh",[["mid","mid"],["side","side"],["left","left"],["right","right"]],dv.ch,k=>{dv.ch=k;renderViewer()});$("dvCh").style.display=stereo?"none":"";
  seg("dvSrc",[["note","the note"],["carrier",`+${d.shift_semitones} st carrier`]],dv.src,k=>{dv.src=k;renderViewer()});
  const sc=$("dvScale");sc.innerHTML="";sc.append(stereo?stereoScale(dv.show):levelScale());
  $("dvAbout").textContent=PIC_ABOUT[pic.mode]+SHOW_ABOUT[dv.show];
  const v=dv.view||fullView();$("dvZoom").textContent=dv.view?`zoom: ${(v.t0*1000).toFixed(1)}–${(v.t1*1000).toFixed(1)} ms · ${fmtHz(midiHz(v.m0))} – ${fmtHz(midiHz(v.m1))}`:"whole note · drag to zoom";
  sharpMap($("dv"),{zoom:true,ex:dv.ex,src:dv.src,mode:pic.mode,show:dv.show,ch:dv.ch,view:dv.view,example:ex[dv.ex],info:d,status:t=>$("dvStatus").textContent=t,onzoom:nv=>{dv.view=nv;renderViewer()}})}
function renderEncoding(){
  const d=enc.data,S=$("encSection"),E=$("encoding");if(!d||!d.examples){S.style.display="none";return}S.style.display="";
  const ex=d.examples,trip=t=>t==null?"exact 0 (mono source)":t.toFixed(0)+" dB";
  let html=`<p class=muted style="margin:0 0 8px">${d.encoding}: ${d.rows} semitone rows (16 Hz – Nyquist) × ${d.frames} frames of ${d.hop_ms} ms at ${ex[0].stored_rate/1000} kHz, mid + side; ${d.spec.redundancy}× the sample count. Every cell holds that band's exact complex samples (1 to 256 per 10 ms), so the pictures below are drawn from all of them at your screen's resolution. Hover any picture for frequency, time and value.</p>
  <div class=dvbar><b>Picture</b><span class=seg id=picSeg></span><span class=muted id=picAbout></span></div><div class=enc3>`;
  ex.forEach(e=>{const r=e.round_trip_db;html+=`<div class=card><b>${e.label}</b><br>${e.note} (MIDI ${e.midi}, ${e.fundamental_hz} Hz) · ${e.family}<br>
    source <b>${(e.source_rate/1000).toFixed(1)} kHz</b> → stored at ${e.stored_rate/1000} kHz · ${e.side_to_mid_db<-200?"mono (side = 0)":"side/mid "+e.side_to_mid_db+" dB"}<br>
    round trip, float32: L ${trip(r.float32.left)} · R ${trip(r.float32.right)} · side ${trip(r.float32.side)}<br>
    <span class=muted>float64: L ${trip(r.float64.left)} · R ${trip(r.float64.right)} · side ${trip(r.float64.side)}</span></div>`});
  html+=`</div><div class=encrow><h3>Waveform (left, right)</h3><p>The 1.2 s note at 192 kHz, as stored for training.</p><div class=enc3>${ex.map((e,i)=>`<canvas class=wave id=wave${i}></canvas>`).join("")}</div></div>`;
  html+=`<div class=encrow><h3>Detail viewer</h3><p id=dvAbout></p><div class=dvbar><span class=seg id=dvEx></span><span class=seg id=dvShow></span><span class=seg id=dvCh></span><span class=seg id=dvSrc></span>
    <button id=dvReset>Reset zoom</button><span class=muted id=dvZoom></span><span class=muted id=dvStatus></span></div><canvas id=dv class=dv></canvas><div id=dvScale></div></div>`;
  html+=`<div class=encrow><h3>Multirate cell layout (same for every note)</h3><p>How many exact complex samples each row stores per 10 ms: 1 in the bass, doubling each octave up to 256 at the top. Nothing is pooled away.</p><canvas class=wave id=cells style="height:110px"></canvas></div>`;
  const sharpRows=[
    ["level_mid",{show:"level",ch:"mid"},"Wavelet image, mid","Mid = (L+R)/2, drawn from the exact samples in the picture chosen above."],
    ["level_side",{show:"level",ch:"side"},"Wavelet image, side","Side = (L−R)/2, same picture. Empty for mono sources: stereo is stored exactly, not estimated."],
    ["carrier",{show:"level",ch:"mid",src:"carrier"},`Key change: moved +${d.shift_semitones} semitones (carrier)`,"Each row moved up by the interval with the exact heterodyne phase ramp and resampled to its new cell size: what the model reshapes into the target note."],
    ["stereo_pan",{show:"pan"},"Stereo image: pan","Where each partial sits between left and right (L = mid + side, R = mid − side), per point in time and frequency; brightness = level."],
    ["stereo_corr",{show:"corr"},"Stereo image: phase correlation","In phase (mono-compatible), unrelated (wide: unison, chorus, room) or opposite phase (cancels when summed to mono)."]];
  sharpRows.forEach(([k,,title,about])=>{html+=`<div class=encrow><h3>${title}</h3><p>${about}</p><div class=enc3>${ex.map((e,i)=>`<canvas class=hm id="sm_${k}_${i}"></canvas>`).join("")}</div><div id="sc_${k}"></div></div>`});
  html+=`<div class=encrow><h3>Stereo image: goniometer</h3><p>Every sample of the note as a point: vertical = mid, horizontal = side. A vertical line is mono, the diagonals are hard left and right, a round cloud is wide and decorrelated, a horizontal line is out of phase.</p>
    <div class=enc3>${ex.map((e,i)=>`<div><canvas class=gonio id=go${i}></canvas>${corrMeter(e)}</div>`).join("")}</div></div>`;
  html+=`<div class=encrow><h3>Stereo image by frequency</h3><p>Per semitone row over the whole note, from the exact cells. Rows more than 60 dB below the loudest row are not drawn.</p>
    <div class=enc3>${ex.map((e,i)=>`<canvas class=wave id=sf${i} style="height:150px"></canvas>`).join("")}</div>
    <div class=legend><span><i style="background:${COLORS[2]}"></i>phase correlation (+1 mono, 0 unrelated, −1 opposite)</span><span><i style="background:${COLORS[4]}"></i>balance (−1 left, +1 right)</span></div></div>`;
  const keys=["if_deviation","phase_coherence","baseband_re","baseband_im","modulation_depth","stereo_width"];
  html+=`<div class=encrow><h3 style="font-size:14px">Feature planes the backbone reads</h3><p>These are per cell by design (one pixel = one 10 ms × semitone cell), so they stay blocky: they are the backbone's input grid, computed from the exact samples. The lossless path additionally embeds every sample of every cell.</p></div>`;
  keys.forEach(k=>{const p=ex[0].planes[k];html+=`<div class=encrow><h3>${p.label}</h3><p>${p.about}</p><div class=enc3>${ex.map((e,i)=>`<canvas class=hm id="hm_${k}_${i}"></canvas>`).join("")}</div><div id="sc_${k}"></div></div>`});
  html+=`<div class=encrow><h3>What the model receives</h3><p>${d.model_inputs}</p></div>`;
  E.innerHTML=html;
  requestAnimationFrame(()=>{ex.forEach((e,i)=>waveform($("wave"+i),e));cellLayout($("cells"),d);
    seg("picSeg",PICTURES,pic.mode,k=>{pic.mode=k;renderEncoding()});$("picAbout").textContent="applies to the viewer and every map down to the feature planes";
    renderViewer();$("dvReset").onclick=()=>{dv.view=null;renderViewer()};
    sharpRows.forEach(([k,cfg])=>{ex.forEach((e,i)=>sharpMap($(`sm_${k}_${i}`),{...cfg,mode:pic.mode,ex:i,example:e,info:d,placeholder:placeholderFrom(e.planes[k],d)}));
      $("sc_"+k).append(cfg.show!=="level"?stereoScale(cfg.show):levelScale())});
    ex.forEach((e,i)=>{gonio($("go"+i),e,i);stereoByFreq($("sf"+i),e,d)});
    keys.forEach(k=>{ex.forEach((e,i)=>heatmap($(`hm_${k}_${i}`),e.planes[k],d,e));$("sc_"+k).append(scaleBar(ex[0].planes[k]))})});
}
async function loadEncoding(){if(enc.run===state.run)return;enc.run=state.run;enc.data=await getJSON("/api/encoding?name="+encodeURIComponent(state.run));renderEncoding()}
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>renderEncoding());
async function load(){const runs=await getJSON("/api/runs");if(!runs.length)return;state.runs=runs;
  const live=runs.find(r=>r.running),wanted=state.run||new URLSearchParams(location.search).get("run")||(live||runs[0]).name;
  const cur=runs.find(r=>r.name===wanted)||live||runs[0];state.run=cur.name;
  const families=[...new Set(runs.map(r=>r.family))];
  $("family").innerHTML=families.map(f=>`<option value="${f}">${f}${runs.some(r=>r.family===f&&r.running)?"  ● live":""}</option>`).join("");$("family").value=cur.family;
  $("run").innerHTML=runs.filter(r=>r.family===cur.family).map(r=>`<option value="${r.name}">${r.version}${r.running?"  ● live":""}</option>`).join("");$("run").value=cur.name;
  const go=$("golive");if(live&&!cur.running){go.style.display="";go.textContent=`Go to live training: ${live.family} ${live.version}`;go.onclick=()=>{state.run=live.name;load()}}else go.style.display="none";
  state.data=await getJSON("/api/run?name="+encodeURIComponent(state.run));render();loadEncoding()}
async function control(action){const r=await fetch(`/api/control?name=${encodeURIComponent(state.run)}&action=${action}`,{method:"POST"});
  const j=await r.json();if(!j.ok)alert(j.message);setTimeout(load,1500)}
$("pausebtn").onclick=()=>{if(confirm(`Pause ${state.run}? It saves its exact state after the current step and stops.`))control("pause")};
$("resumebtn").onclick=()=>{if(confirm(`Resume training ${state.run}? This starts the training process again.`))control("resume")};
$("family").onchange=e=>{const first=(state.runs||[]).find(r=>r.family===e.target.value);if(first){state.run=first.name;load()}};
$("run").onchange=e=>{state.run=e.target.value;load()};$("refresh").onclick=load;$("probeFeature").onchange=renderProbes;$("evalSel").onchange=renderEvals;
window.addEventListener("resize",()=>state.data&&render());load();setInterval(load,30000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, kind="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        runs = discover_runs()
        if url.path == "/":
            return self._send(PAGE, "text/html")
        if url.path == "/api/runs":
            return self._send(json.dumps(run_index()))
        if url.path == "/api/encoding":
            name = parse_qs(url.query).get("name", [""])[0]
            path = runs[name] / "encoding_examples.json" if name in runs else None
            return self._send(path.read_text(encoding="utf-8") if path and path.exists() else "{}")
        if url.path in ("/api/encoding/view", "/api/encoding/scope", "/api/demo/view"):
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if q.get("name") not in runs:
                self.send_error(404, "unknown run")
                return
            try:
                fn = VIEWS.demo if url.path == "/api/demo/view" else VIEWS.view if url.path.endswith("view") else VIEWS.scope
                return self._send(fn(runs[q["name"]], q), "application/octet-stream")
            except (ImportError, OSError) as error:
                self.send_error(503, f"views need numpy, torch and wavelet2vec: {error}")
            except ValueError as error:
                self.send_error(400, str(error))
            return
        if url.path == "/api/file":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if q.get("name") not in runs:
                self.send_error(404, "unknown run")
                return
            folder = runs[q["name"]].resolve()
            path = (folder / q.get("path", "")).resolve()
            if folder not in path.parents or path.suffix.lower() not in AUDIO_TYPES or not path.is_file():
                self.send_error(404, "no such audio file in this run")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", AUDIO_TYPES[path.suffix.lower()])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return
        if url.path == "/api/run":
            name = parse_qs(url.query).get("name", [""])[0]
            if name not in runs:
                self.send_error(404, "unknown run")
                return
            return self._send(json.dumps(run_payload(name, runs[name]), default=float))
        self.send_error(404)

    def do_POST(self):
        """Run control (local only): ``/api/control?name=<run>&action=pause|resume``."""
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        runs = discover_runs()
        if url.path != "/api/control" or q.get("name") not in runs or q.get("action") not in ("pause", "resume"):
            self.send_error(404)
            return
        folder = runs[q["name"]]
        try:
            if q["action"] == "pause":
                if not ctl.running_pid(folder):
                    raise RuntimeError("the run is not training")
                ctl.request_pause(folder)
                reply = {"ok": True, "message": "pause requested: the run saves its exact state after the current step"}
            else:
                reply = {"ok": True, "pid": ctl.resume(folder), "message": "resumed"}
            _TRAINING["time"] = 0.0
        except Exception as error:
            reply = {"ok": False, "message": str(error)}
        return self._send(json.dumps(reply))

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Training dashboard on http://localhost:{args.port}  (runs: {', '.join(discover_runs())})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
