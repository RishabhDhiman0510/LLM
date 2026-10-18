import argparse
import shutil
from pathlib import Path

import torch

from wikigpt.checkpoint import load_checkpoint
from wikigpt.config import ModelConfig, TrainConfig, add_config_args, config_from_args
from wikigpt.data import TokenDataset
from wikigpt.generate import generate
from wikigpt.model import GPT
from wikigpt.tokenizer import Tokenizer
from wikigpt.trainer import Trainer

PROMPTS = ["The history of", "In mathematics, a", "The city of", "During the Second World War,"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain WikiGPT on encoded Wikipedia tokens.")
    parser.add_argument("--data", type=Path, required=True, help="folder with train.bin, val.bin and tokenizer.json")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/runs/pretrain"))
    parser.add_argument("--resume", type=Path, help="ckpt.pt from an earlier run")
    add_config_args(parser, ModelConfig(), skip=("vocab_size",))
    add_config_args(parser, TrainConfig())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.load(args.data / "tokenizer.json")
    train_cfg = config_from_args(TrainConfig, args)
    ckpt = load_checkpoint(args.resume) if args.resume else None
    if ckpt:
        model_cfg = ModelConfig.from_dict(ckpt["model_config"])
    else:
        model_cfg = config_from_args(ModelConfig, args, vocab_size=tokenizer.vocab_size)

    torch.manual_seed(train_cfg.seed)
    model = GPT(model_cfg)
    print(f"parameters: {model.num_params():,}")

    # a new seed after a resume so the run does not replay the same windows
    seed = train_cfg.seed + (ckpt["step"] if ckpt else 0)
    datasets = {
        "train": TokenDataset(args.data / "train.bin", model_cfg.context_length, seed),
        "val": TokenDataset(args.data / "val.bin", model_cfg.context_length, seed + 1),
    }

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        return datasets[split].batch(train_cfg.batch_size)

    def sample(model: GPT) -> str:
        texts = []
        for prompt in PROMPTS:
            ids = generate(model, tokenizer.encode(prompt), max_new_tokens=60, stop_ids={tokenizer.eot_id})
            texts.append(prompt + tokenizer.decode(ids))
        return "\n\n".join(texts)

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.data / "tokenizer.json", args.out / "tokenizer.json")
    trainer = Trainer(model, train_cfg, get_batch, args.out, sample_fn=sample)
    if ckpt:
        trainer.restore(ckpt, previous_log=args.resume.parent / "log.csv")
    trainer.train()


if __name__ == "__main__":
    main()
