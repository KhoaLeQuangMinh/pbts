"""Run logger. Local files are the master copy; W&B (private project) is a live mirror.

All files are append-only JSON lines, so a resumed run continues the same logs:
  provenance/            written once at start (and on every resume, as provenance/resume_<n>/)
  steps.jsonl            every LOG_EVERY_STEPS steps
  epochs.jsonl           every epoch
  case_losses.jsonl      per-training-case loss, every epoch
  monitor.jsonl          fast scores on the fixed monitor cases, every MONITOR_EVERY epochs (+ monitor_png/)
  system.jsonl           GPU/CPU/RAM every SYSTEM_EVERY_S seconds (background thread)
  events.jsonl           start / resume / checkpoint / requeue / NaN / crash / rescue checks
"""
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from pbts import config as C

LOG_EVERY_STEPS = 50
MONITOR_EVERY = 5
SYSTEM_EVERY_S = 30


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


class RunLogger:
    def __init__(self, run_dir, run_config: dict, use_wandb: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = run_config
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sys_thread = None
        self.wandb = None
        if use_wandb:
            self._init_wandb()

    # ------------------------------------------------------------------ files
    def _append(self, name, rec):
        rec = {"time": _now(), **rec}
        with self._lock, open(self.dir / name, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def step(self, rec):
        self._append("steps.jsonl", rec)
        self._wb({f"step/{k}": v for k, v in rec.items() if isinstance(v, (int, float))}, rec.get("global_step"))

    def epoch(self, rec):
        self._append("epochs.jsonl", rec)
        self._wb({f"epoch/{k}": v for k, v in rec.items() if isinstance(v, (int, float))}, rec.get("global_step"))

    def case_losses(self, epoch, losses: dict):
        self._append("case_losses.jsonl", {"epoch": epoch, "losses": losses})

    def monitor(self, rec, images=()):
        self._append("monitor.jsonl", rec)
        flat = {}
        for r in rec.get("rows", []):
            for k, v in r.items():
                if isinstance(v, (int, float)) and k not in ("epoch",):
                    flat[f"monitor/{r['region']}/{k}"] = v
        for k, v in rec.get("summary", {}).items():
            flat[f"monitor/{k}"] = v
        if self.wandb is not None and C.WANDB_LOG_IMAGES and images:
            import wandb
            flat.update({f"monitor_img/{Path(p).stem.split('_', 2)[-1]}": wandb.Image(str(p)) for p in images})
        self._wb(flat, rec.get("global_step"))

    def event(self, kind, **kw):
        self._append("events.jsonl", {"event": kind, **kw})
        if self.wandb is not None:
            try:
                self.wandb.log({"events/" + kind: 1}, commit=False)
            except Exception:
                pass

    # ------------------------------------------------------------------ provenance
    def provenance(self, extra: dict = None):
        n = len(list(self.dir.glob("provenance*")))
        pdir = self.dir / ("provenance" if n == 0 else f"provenance/resume_{n}")
        pdir.mkdir(parents=True, exist_ok=True)
        json.dump(self.cfg, open(pdir / "run_config.json", "w"), indent=1, default=_json_default)
        ver = C.ROOT / "code/VERSION.json"
        if ver.exists():
            (pdir / "code_version.json").write_text(ver.read_text())
            diff = C.ROOT / "code/VERSION.diff"
            if diff.exists():
                (pdir / "code_uncommitted.diff").write_text(diff.read_text())
        else:
            self.event("warning", msg="code/VERSION.json missing: code not synced with scripts/sync.sh")
        for name, cmd in [("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
                          ("nvidia_smi.txt", ["nvidia-smi", "-q"])]:
            try:
                (pdir / name).write_text(subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout)
            except Exception as e:
                (pdir / name).write_text(f"failed: {e!r}")
        host = {"hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
                "argv": sys.argv, "cwd": os.getcwd(), "pid": os.getpid(),
                "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        try:
            import torch
            host.update({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         "gpu_total_GB": torch.cuda.get_device_properties(0).total_memory / 2**30
                         if torch.cuda.is_available() else None})
        except Exception:
            pass
        json.dump(host, open(pdir / "host.json", "w"), indent=1, default=_json_default)
        if extra:
            for name, obj in extra.items():
                json.dump(obj, open(pdir / name, "w"), indent=1, default=_json_default)
        self.event("provenance", dir=str(pdir))

    # ------------------------------------------------------------------ system monitor
    def start_system_monitor(self):
        if self._sys_thread is not None:
            return
        self._sys_thread = threading.Thread(target=self._sys_loop, daemon=True)
        self._sys_thread.start()

    def _sys_loop(self):
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            pynvml, handles = None, []
        proc = psutil.Process()
        t0 = time.time()
        while not self._stop.wait(SYSTEM_EVERY_S):
            rec = {"uptime_s": round(time.time() - t0), "cpu_percent": psutil.cpu_percent(), "ram_used_GB": psutil.virtual_memory().used / 2**30,
                   "proc_rss_GB": proc.memory_info().rss / 2**30,
                   "children_rss_GB": sum(c.memory_info().rss for c in proc.children(recursive=True)) / 2**30}
            try:
                import torch
                if torch.cuda.is_available():
                    rec["torch_alloc_GB"] = torch.cuda.memory_allocated() / 2**30
                    rec["torch_reserved_GB"] = torch.cuda.memory_reserved() / 2**30
            except Exception:
                pass
            for i, h in enumerate(handles):
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    rec[f"gpu{i}_util"], rec[f"gpu{i}_mem_used_GB"] = u.gpu, m.used / 2**30
                except Exception:
                    pass
            self._append("system.jsonl", rec)
            self._wb({f"system/{k}": v for k, v in rec.items()}, None)

    # ------------------------------------------------------------------ W&B
    def _init_wandb(self):
        try:
            import wandb
            os.environ.setdefault("WANDB_API_KEY", C.WANDB_KEY_FILE.read_text().strip())
            entity = C.WANDB_ENTITY_FILE.read_text().strip()
            id_file = self.dir / "wandb_id.txt"
            # own id (wandb.util.generate_id no longer exists in wandb 0.31; every run before 2026-11-20 fell
            # back to local-only logging because of it, and is mirrored afterwards by wandb_backfill)
            run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:12]
            id_file.write_text(run_id)
            self.wandb = wandb.init(entity=entity, project=C.WANDB_PROJECT, id=run_id, resume="allow",
                                    name=self.cfg.get("run_name"), group=self.cfg.get("model"),
                                    job_type=self.cfg.get("tag"), config=self.cfg, dir=str(self.dir))
            self.wandb.define_metric("global_step")
            for pat in ["step/*", "epoch/*", "monitor/*", "monitor_img/*", "events/*"]:
                self.wandb.define_metric(pat, step_metric="global_step")
            self.wandb.define_metric("system/uptime_s")
            self.wandb.define_metric("system/*", step_metric="system/uptime_s")
        except Exception as e:
            self.wandb = None
            self._append("events.jsonl", {"event": "wandb_unavailable", "error": repr(e)[:300]})

    def _wb(self, d, step):
        if self.wandb is None or not d:
            return
        if step is not None:
            d = {**d, "global_step": step}
        try:
            self.wandb.log(d)
        except Exception:
            pass

    def close(self, exit_code=0):
        self._stop.set()
        if self.wandb is not None:
            try:
                self.wandb.finish(exit_code=exit_code)
            except Exception:
                pass


def _jsonl_safe(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:      # a line cut by a crash/requeue must not break replay
            pass
    return out


def log_crash(logger: "RunLogger", exc: BaseException, **ctx):
    logger.event("crash", error=repr(exc), traceback=traceback.format_exc(), **ctx)
