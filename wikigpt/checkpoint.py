import copy
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig
from .model import GPT
from .tokenizer import Tokenizer

BUNDLE_FILES = ("model.pt", "config.json", "tokenizer.json")


def save_checkpoint(path: Path, model: GPT, optimizer, scaler, step: int, train_config) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "model_config": model.cfg.to_dict(),
        "train_config": asdict(train_config),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    # replace in one step so a crash never leaves a half-written checkpoint
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def model_from_checkpoint(ckpt: dict, **overrides) -> GPT:
    model = GPT(ModelConfig.from_dict({**ckpt["model_config"], **overrides}))
    model.load_state_dict(ckpt["model"])
    return model


def export_bundle(model: GPT, tokenizer_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    half = copy.deepcopy(model).cpu().half()
    torch.save(half.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
    shutil.copy(tokenizer_path, out_dir / "tokenizer.json")
    return out_dir


def load_bundle(bundle_dir: Path, device: str = "cpu") -> tuple[GPT, Tokenizer]:
    bundle_dir = Path(bundle_dir)
    missing = [name for name in BUNDLE_FILES if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"model bundle in {bundle_dir} is missing: {', '.join(missing)}")
    cfg = ModelConfig.from_dict(json.loads((bundle_dir / "config.json").read_text()))
    model = GPT(cfg)
    model.load_state_dict(torch.load(bundle_dir / "model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, Tokenizer.load(bundle_dir / "tokenizer.json")
