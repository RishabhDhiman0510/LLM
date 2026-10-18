import csv
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .checkpoint import save_checkpoint
from .config import TrainConfig
from .model import GPT

BatchFn = Callable[[str], tuple[torch.Tensor, torch.Tensor]]
SampleFn = Callable[[GPT], str]
LOG_FIELDS = ["step", "train_loss", "val_loss", "lr", "tokens_per_sec", "elapsed_min"]


def build_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def read_log(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_log(path: Path, rows: list[dict], append: bool = True) -> None:
    new = not append or not path.exists()
    with path.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def plot_log(log_path: Path, png_path: Path) -> None:
    rows = read_log(log_path)
    train = [(int(r["step"]), float(r["train_loss"])) for r in rows]
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(*zip(*train), label="train", linewidth=1)
    if val:
        ax.plot(*zip(*val), label="val", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


class Trainer:
    def __init__(self, model: GPT, cfg: TrainConfig, get_batch: BatchFn, out_dir: Path, sample_fn: SampleFn | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.cfg = cfg
        self.get_batch = get_batch
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.csv"
        self.sample_fn = sample_fn
        self.step = 0
        self.use_amp = self.device == "cuda"
        self.optimizer = build_optimizer(self.model, cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.forward_model = self._compiled()

    def _compiled(self) -> torch.nn.Module:
        # triton needs compute capability 7.0+, so older GPUs such as the P100 run eagerly
        if self.cfg.compile and self.use_amp and torch.cuda.get_device_capability()[0] >= 7:
            return torch.compile(self.model)
        return self.model

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def restore(self, ckpt: dict, previous_log: Path | None = None) -> None:
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        # keep only log rows up to the checkpoint so resumed curves do not overlap
        if previous_log and previous_log.exists():
            rows = [r for r in read_log(previous_log) if int(r["step"]) <= self.step]
            write_log(self.log_path, rows, append=False)

    def train(self) -> None:
        cfg = self.cfg
        print(f"training on {self.device} from step {self.step} to {cfg.max_steps}")
        start = time.time()
        window_start, window_tokens, window_loss, window_steps = time.time(), 0, 0.0, 0
        while self.step < cfg.max_steps:
            lr = lr_at(self.step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            step_loss, step_tokens = self._train_step()
            self.step += 1
            window_loss += step_loss
            window_tokens += step_tokens
            window_steps += 1

            last = self.step == cfg.max_steps
            should_eval = self.step % cfg.eval_interval == 0 or last
            if should_eval or self.step % cfg.log_interval == 0:
                rate = window_tokens / (time.time() - window_start)
                val_loss = self.evaluate() if should_eval else None
                self._log(window_loss / window_steps, val_loss, lr, rate, time.time() - start)
                window_start, window_tokens, window_loss, window_steps = time.time(), 0, 0.0, 0
            if self.step % cfg.ckpt_interval == 0 or last:
                self.save()
                self._sample()
        if self.log_path.exists():
            plot_log(self.log_path, self.out_dir / "loss.png")

    def _train_step(self) -> tuple[float, int]:
        self.forward_model.train()
        total = torch.zeros((), device=self.device)
        tokens = 0
        for _ in range(self.cfg.grad_accum):
            x, y = (t.to(self.device, non_blocking=True) for t in self.get_batch("train"))
            with self._autocast():
                _, loss = self.forward_model(x, y)
            self.scaler.scale(loss / self.cfg.grad_accum).backward()
            total += loss.detach()
            tokens += x.numel()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return total.item() / self.cfg.grad_accum, tokens

    @torch.no_grad()
    def evaluate(self) -> float:
        self.forward_model.eval()
        total = 0.0
        for _ in range(self.cfg.eval_batches):
            x, y = (t.to(self.device) for t in self.get_batch("val"))
            with self._autocast():
                _, loss = self.forward_model(x, y)
            total += loss.item()
        self.forward_model.train()
        return total / self.cfg.eval_batches

    def save(self) -> None:
        save_checkpoint(self.out_dir / "ckpt.pt", self.model, self.optimizer, self.scaler, self.step, self.cfg)

    def _sample(self) -> None:
        if self.sample_fn is None:
            return
        text = self.sample_fn(self.model)
        self.model.train()
        with (self.out_dir / "samples.txt").open("a", encoding="utf-8") as f:
            f.write(f"--- step {self.step} ---\n{text}\n\n")
        print(text, flush=True)

    def _log(self, train_loss: float, val_loss: float | None, lr: float, rate: float, elapsed: float) -> None:
        row = {
            "step": self.step,
            "train_loss": f"{train_loss:.4f}",
            "val_loss": "" if val_loss is None else f"{val_loss:.4f}",
            "lr": f"{lr:.3e}",
            "tokens_per_sec": f"{rate:.0f}",
            "elapsed_min": f"{elapsed / 60:.1f}",
        }
        write_log(self.log_path, [row])
        val = "" if val_loss is None else f" | val {val_loss:.4f} (ppl {math.exp(val_loss):.1f})"
        print(
            f"step {self.step}/{self.cfg.max_steps} | train {train_loss:.4f}{val} | lr {lr:.2e} "
            f"| {rate / 1000:.1f}k tok/s | {elapsed / 60:.1f} min",
            flush=True,
        )
