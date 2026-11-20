"""PBTSTrainerMixin: put in front of any nnU-Net trainer class (class X(PBTSTrainerMixin, SomeNnUNetTrainer)).

What it changes (everything else is the parent trainer's own behaviour, i.e. the model's recipe):
  - training length in samples (1 epoch = 1 pass over the training fold), from the run config
  - optimiser lr / weight decay / gradient clipping from pbts.protocol.RECIPES (lr x rescue factor)
  - data loader pads with each case's background value (pbts.train.loader)
  - no 'best' checkpoint; weights saved at 25/50/75/100% of training + 'latest' every 30 min
  - exhaustive logging (pbts.logging.runlog), monitor every MONITOR_EVERY epochs
  - guards: non-finite loss -> stop + save batch info; SIGUSR1 (SLURM time limit) -> checkpoint + requeue
  - protocol rescue check 'not_learning' at 10% of training (first run of each model only)
"""
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import autocast

from pbts import config as C
from pbts import protocol as P
from pbts.data.guard import check_not_locked
from pbts.logging.runlog import LOG_EVERY_STEPS, MONITOR_EVERY, RunLogger, log_crash
from pbts.train.loader import PBTSDataLoader
from pbts.train.monitor import run_monitor

LATEST_EVERY_S = 30 * 60
FRACTIONS = (25, 50, 75, 100)


class NonFiniteLoss(RuntimeError):
    pass


class PBTSTrainerMixin:
    recipe_name: str = None

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.run_cfg = json.load(open(os.environ["PBTS_RUN_CONFIG"]))
        self.recipe = P.RECIPES[self.recipe_name]
        self.initial_lr = self.recipe["lr"] * float(self.run_cfg.get("lr_factor", 1.0))
        self.weight_decay = self.recipe["weight_decay"]
        gc = self.recipe.get("grad_clip")
        self.grad_clip = float(gc) if gc is not None else math.inf       # None in the recipe = no clipping
        self.save_every = 10 ** 9                    # nnU-Net's periodic save disabled; we manage checkpoints
        self.pbts_log = None
        self.global_step = 0
        self._requeue = False
        self._last_latest = time.time()
        self._epoch_buf = {}

    # ------------------------------------------------------------------ setup
    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample()
        bs = self.run_cfg.get("batch_size") or self.recipe.get("batch_size")
        if bs:
            self.batch_size = int(bs)
        n_tr, n_val = len(self.run_cfg["train_cases"]), len(self.run_cfg["val_cases"])
        self.num_iterations_per_epoch = math.ceil(n_tr / self.batch_size)
        self.num_epochs = max(1, math.ceil(self.run_cfg["samples"] / (self.num_iterations_per_epoch * self.batch_size)))
        self.num_val_iterations_per_epoch = math.ceil(n_val / self.batch_size)
        smoke = self.run_cfg.get("smoke")
        if smoke:
            self.num_iterations_per_epoch, self.num_epochs = smoke["iters"], smoke["epochs"]
            self.num_val_iterations_per_epoch = 2
        if hasattr(self, "warmup_duration_whole_net"):          # warmup trainers (Primus): same fraction
            self.warmup_duration_whole_net = max(1, round(self.recipe.get("warmup_fraction", 0.0) * self.num_epochs))
        self._frac_epoch = {f: max(1, round(f / 100 * self.num_epochs)) - 1 for f in FRACTIONS}
        # prediction uses the training input size: whole-head frame, or sliding 128^3 windows for the pilot
        self.frame_xyz = tuple(reversed(self.configuration_manager.patch_size))

    def get_dataloaders(self):
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as T
        orig = T.nnUNetDataLoader
        T.nnUNetDataLoader = PBTSDataLoader
        try:
            return super().get_dataloaders()
        finally:
            T.nnUNetDataLoader = orig

    def do_split(self):
        tr, val = super().do_split()
        if sorted(tr) != sorted(self.run_cfg["train_cases"]) or sorted(val) != sorted(self.run_cfg["val_cases"]):
            raise RuntimeError("nnU-Net split differs from the study split for this fold")
        check_not_locked(list(tr) + list(val))
        return tr, val

    def configure_optimizers(self, *args, **kwargs):
        out = super().configure_optimizers(*args, **kwargs)
        opt = out[0]
        for g in opt.param_groups:                               # enforce protocol weight decay
            g["weight_decay"] = self.weight_decay
        return out

    def _model_summary(self):
        n = sum(p.numel() for p in self.network.parameters())
        nt = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        return {"params": n, "trainable_params": nt, "class": type(self.network).__name__,
                "batch_size": self.batch_size, "iters_per_epoch": self.num_iterations_per_epoch,
                "num_epochs": self.num_epochs, "val_iters_per_epoch": self.num_val_iterations_per_epoch,
                "initial_lr": self.initial_lr, "weight_decay": self.weight_decay, "grad_clip": self.grad_clip,
                "deep_supervision": bool(self.enable_deep_supervision),
                "warmup_epochs": getattr(self, "warmup_duration_whole_net", 0),
                "fraction_checkpoint_epochs": self._frac_epoch,
                "patch_size_zyx": list(self.configuration_manager.patch_size),
                "optimizer": repr(self.optimizer), "lr_scheduler": type(self.lr_scheduler).__name__,
                "loss": repr(self.loss)}

    def on_train_start(self):
        resumed = self.current_epoch > 0
        super().on_train_start()
        self.pbts_log = RunLogger(self.run_cfg["run_dir"], self.run_cfg, use_wandb=not self.run_cfg.get("no_wandb"))
        self.pbts_log.provenance(extra={"model_summary.json": self._model_summary(),
                                        "nnunet_plans.json": self.plans_manager.plans,
                                        "recipe.json": self.recipe,
                                        "shared_training.json": P.SHARED_TRAINING,
                                        "rescue_rule.json": P.RESCUE})
        self.pbts_log.start_system_monitor()
        self.global_step = self.current_epoch * self.num_iterations_per_epoch
        self.pbts_log.event("resume" if resumed else "train_start", epoch=self.current_epoch,
                            num_epochs=self.num_epochs, global_step=self.global_step)
        signal.signal(signal.SIGUSR1, self._on_sigusr1)

    def _on_sigusr1(self, *_):
        self._requeue = True

    # ------------------------------------------------------------------ one step
    def _loss_parts(self, output, target):
        inner = getattr(self.loss, "loss", self.loss)            # unwrap DeepSupervisionWrapper
        out0 = output[0] if isinstance(output, (list, tuple)) else output
        tgt0 = target[0] if isinstance(target, (list, tuple)) else target
        with torch.no_grad(), autocast(self.device.type, enabled=self.device.type == "cuda"):
            ce = float(inner.ce(out0, tgt0[:, 0]))
            soft_dice = -float(inner.dc(out0, tgt0))       # nnU-Net's dice term is -soft_dice
            per = [float(inner(out0[i:i + 1], tgt0[i:i + 1])) for i in range(out0.shape[0])]
        return ce, soft_dice, per

    def _group_grad_norms(self):
        out = {}
        for name, mod in self.network.named_children():
            gs = [p.grad.detach().float().norm() ** 2 for p in mod.parameters() if p.grad is not None]
            if gs:
                out[name] = float(torch.stack(gs).sum().sqrt())
        return out

    def train_step(self, batch: dict) -> dict:
        data, target = batch["data"].to(self.device, non_blocking=True), batch["target"]
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            output = self.network(data)
            l = self.loss(output, target)
        if not torch.isfinite(l):
            raise NonFiniteLoss(f"non-finite loss {float(l)} on cases {list(batch['keys'])}")
        ce, soft_dice, per = self._loss_parts(output, target)
        log_groups = (self.global_step + 1) % LOG_EVERY_STEPS == 0
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            groups = self._group_grad_norms() if log_groups else {}
            gn = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            scale = float(self.grad_scaler.get_scale())
        else:
            l.backward()
            groups = self._group_grad_norms() if log_groups else {}
            gn = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
            scale = 1.0
        return {"loss": float(l.detach()), "ce": ce, "soft_dice": soft_dice, "grad_norm": float(gn), "amp_scale": scale,
                "grad_groups": groups, "keys": list(batch["keys"]), "per_sample": per}

    # ------------------------------------------------------------------ loop
    def run_training(self):
        self.on_train_start()
        try:
            for _ in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()
                self.on_train_epoch_start()
                outs, t_epoch = [], time.time()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(self.num_iterations_per_epoch):
                    t0 = time.time()
                    batch = next(self.dataloader_train)
                    t1 = time.time()
                    o = self.train_step(batch)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    o["data_s"], o["step_s"] = t1 - t0, time.time() - t1
                    self.global_step += 1
                    outs.append(o)
                    if self.global_step % LOG_EVERY_STEPS == 0:
                        recent = outs[-LOG_EVERY_STEPS:]
                        self.pbts_log.step({
                            "global_step": self.global_step, "epoch": self.current_epoch,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            **{k: float(np.mean([r[k] for r in recent])) for k in
                               ("loss", "ce", "soft_dice", "grad_norm", "data_s", "step_s")},
                            "grad_norm_max": float(np.max([r["grad_norm"] for r in recent])),
                            "amp_scale": o["amp_scale"], **{f"gradnorm_{k}": v for k, v in o["grad_groups"].items()},
                            "gpu_peak_GB": torch.cuda.max_memory_allocated() / 2**30})
                self._epoch_buf = {"outs": outs, "epoch_s": time.time() - t_epoch}
                self.on_train_epoch_end(outs)
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    vals = [self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)]
                    self.on_validation_epoch_end(vals)
                self.on_epoch_end()
                if self._requeue and self.current_epoch < self.num_epochs:   # finish the epoch, then requeue
                    self._checkpoint_and_requeue()
        except NonFiniteLoss as e:
            self.pbts_log.event("non_finite_loss", error=str(e), epoch=self.current_epoch, global_step=self.global_step)
            self._write_rescue("non_finite_loss", str(e))
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_nonfinite.pth"))
            self.pbts_log.close(exit_code=3)
            raise SystemExit(3)
        except SystemExit:
            raise
        except BaseException as e:
            log_crash(self.pbts_log, e, epoch=self.current_epoch, global_step=self.global_step)
            try:
                self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))
            except Exception:
                pass
            self.pbts_log.close(exit_code=1)
            raise
        self.on_train_end()
        self.pbts_log.event("train_end", epoch=self.current_epoch, global_step=self.global_step)

    def on_train_epoch_end(self, train_outputs):
        self.logger.log("train_losses", float(np.mean([o["loss"] for o in train_outputs])), self.current_epoch)

    def on_epoch_end(self):
        ep = self.current_epoch
        self.logger.log("epoch_end_timestamps", time.time(), ep)
        outs = self._epoch_buf.get("outs", [])
        n_samples = len(outs) * self.batch_size
        dice_val = self.logger.get_value("dice_per_class_or_region", step=-1)
        rec = {"epoch": ep, "global_step": self.global_step, "lr": self.optimizer.param_groups[0]["lr"],
               "train_loss": float(np.mean([o["loss"] for o in outs])),
               "train_ce": float(np.mean([o["ce"] for o in outs])),
               "train_soft_dice": float(np.mean([o["soft_dice"] for o in outs])),
               "val_loss": float(self.logger.get_value("val_losses", step=-1)),
               "ema_pseudo_dice": float(self.logger.get_value("ema_fg_dice", step=-1)),
               **{f"val_pseudo_dice_{n}": float(d) for n, d in zip(["ET", "NET", "CC", "ED"], dice_val)},
               "grad_norm_mean": float(np.mean([o["grad_norm"] for o in outs])),
               "grad_norm_max": float(np.max([o["grad_norm"] for o in outs])),
               # AMP: a step whose fp16 gradients overflow has an infinite norm; GradScaler skips it (weights untouched)
               # and halves the loss scale. Logged explicitly so inf in grad_norm_* is never mistaken for divergence.
               "amp_skipped_steps": int(sum(not math.isfinite(o["grad_norm"]) for o in outs)),
               "grad_norm_mean_finite": float(np.mean([o["grad_norm"] for o in outs if math.isfinite(o["grad_norm"])]
                                                      or [np.nan])),
               "clipped_fraction": float(np.mean([o["grad_norm"] > self.grad_clip for o in outs])),
               "data_wait_s": float(np.sum([o["data_s"] for o in outs])),
               "compute_s": float(np.sum([o["step_s"] for o in outs])),
               "epoch_s": float(self._epoch_buf.get("epoch_s", 0)),
               "samples_per_s": n_samples / max(self._epoch_buf.get("epoch_s", 1e-9), 1e-9),
               "gpu_peak_GB": torch.cuda.max_memory_allocated() / 2**30}
        self.pbts_log.epoch(rec)
        per_case = {}
        for o in outs:
            for k, v in zip(o["keys"], o["per_sample"]):
                per_case.setdefault(k, []).append(v)
        self.pbts_log.case_losses(ep, {k: float(np.mean(v)) for k, v in per_case.items()})
        self.print_to_log_file(f"epoch {ep}: train {rec['train_loss']:.4f} val {rec['val_loss']:.4f} "
                               f"pseudo-dice {[round(float(d), 3) for d in dice_val]} {rec['epoch_s']:.0f}s")

        if (ep + 1) % MONITOR_EVERY == 0 or ep == self.num_epochs - 1:
            self._monitor(ep)
        if self.run_cfg.get("rescue_checks"):
            self._rescue_not_learning(ep)
        for f, e in self._frac_epoch.items():
            if ep == e:
                self._save_weights(f, ep)
        if time.time() - self._last_latest > LATEST_EVERY_S and ep != self.num_epochs - 1:
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))
            self._last_latest = time.time()
            self.pbts_log.event("checkpoint_latest", epoch=ep)
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1

    # ------------------------------------------------------------------ helpers
    def _monitor(self, ep):
        t0 = time.time()
        self.network.eval()
        self.set_deep_supervision_enabled(False)
        try:
            rows, summary, images = run_monitor(self.network, self.run_cfg["monitor_cases"], ep, self.run_cfg["run_dir"],
                                                frame_xyz=self.frame_xyz)
            self.pbts_log.monitor({"epoch": ep, "global_step": self.global_step, "rows": rows, "summary": summary,
                                   "seconds": time.time() - t0}, images)
        finally:
            self.set_deep_supervision_enabled(self.enable_deep_supervision)
            self.network.train()

    def _save_weights(self, frac, ep):
        d = Path(self.run_cfg["run_dir"]) / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        mod = self.network._orig_mod if hasattr(self.network, "_orig_mod") else self.network
        torch.save({"network_weights": mod.state_dict(), "epoch": ep, "global_step": self.global_step,
                    "fraction": frac, "trainer_name": type(self).__name__}, d / f"weights_{frac:03d}.pth")
        self.pbts_log.event("checkpoint_fraction", fraction=frac, epoch=ep)

    def _write_rescue(self, trigger, detail):
        p = Path(self.run_cfg["run_dir"]) / "RESCUE_TRIGGER.json"
        json.dump({"trigger": trigger, "detail": detail, "rule": P.RESCUE["triggers"].get(trigger),
                   "epoch": self.current_epoch, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(p, "w"), indent=1)

    def _rescue_not_learning(self, ep):
        check_ep = max(9, round(0.1 * self.num_epochs) - 1)
        if ep != check_ep:
            return
        losses = [json.loads(l)["train_loss"] for l in open(Path(self.run_cfg["run_dir"]) / "epochs.jsonl")]
        first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
        ok = last < 0.9 * first
        self.pbts_log.event("rescue_check_not_learning", epoch=ep, first5=first, last5=last, passed=ok)
        if not ok:
            self._write_rescue("not_learning", f"mean loss first 5 epochs {first:.4f}, last 5 {last:.4f}")
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))
            self.pbts_log.close(exit_code=3)
            raise SystemExit(3)

    def _checkpoint_and_requeue(self):
        # called after on_epoch_end incremented current_epoch; nnU-Net stores current_epoch + 1, so compensate
        self.current_epoch -= 1
        self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))
        self.current_epoch += 1
        self.pbts_log.event("requeue", epoch=self.current_epoch, global_step=self.global_step)
        self.pbts_log.close(exit_code=0)
        job = os.environ.get("SLURM_JOB_ID")
        if job:
            subprocess.run(["scontrol", "requeue", job], check=False)
        sys.exit(0)

    def perform_actual_validation(self, save_probabilities: bool = False):
        pass                                     # replaced by pbts.train.run (shared predictor + evaluator)
