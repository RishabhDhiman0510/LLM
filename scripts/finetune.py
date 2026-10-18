import argparse
import shutil
from pathlib import Path

import torch

from wikigpt.chat import chat_reply
from wikigpt.checkpoint import load_checkpoint, model_from_checkpoint
from wikigpt.config import FinetuneConfig, add_config_args, config_from_args
from wikigpt.model import GPT
from wikigpt.sft import SFTBatches, build_sft_data
from wikigpt.tokenizer import Tokenizer
from wikigpt.trainer import Trainer

PROMPTS = [
    "Hi, how are you?",
    "What is the capital of France?",
    "Give me three tips for staying healthy.",
    "Explain what a computer is in simple words.",
    "Who wrote Romeo and Juliet?",
    "What is photosynthesis?",
    "Write a short poem about the sea.",
    "What is the difference between a virus and a bacterium?",
    "Can you recommend a good book?",
    "Thank you for your help!",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat-tune a pretrained WikiGPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretrain ckpt.pt with tokenizer.json next to it")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/runs/finetune"))
    add_config_args(parser, FinetuneConfig())
    return parser.parse_args()


def replies(model: GPT, tokenizer: Tokenizer, prompts: list[str]) -> str:
    blocks = []
    for prompt in prompts:
        reply = chat_reply(model, tokenizer, [{"role": "user", "content": prompt}])
        blocks.append(f"> {prompt}\n{reply}")
    return "\n\n".join(blocks)


def main() -> None:
    args = parse_args()
    cfg = config_from_args(FinetuneConfig, args)
    torch.manual_seed(cfg.seed)
    tokenizer_path = args.checkpoint.parent / "tokenizer.json"
    tokenizer = Tokenizer.load(tokenizer_path)
    model = model_from_checkpoint(load_checkpoint(args.checkpoint), dropout=cfg.dropout)

    train, val = build_sft_data(tokenizer, model.cfg.context_length, cfg.seed)
    print(f"chat examples: {len(train):,} train, {len(val):,} val")
    batches = {
        "train": SFTBatches(train, cfg.batch_size, tokenizer.eot_id, cfg.seed),
        "val": SFTBatches(val, cfg.batch_size, tokenizer.eot_id, cfg.seed + 1),
    }
    if cfg.max_steps == 0:
        cfg.max_steps = cfg.epochs * len(batches["train"]) // cfg.grad_accum

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(tokenizer_path, args.out / "tokenizer.json")
    trainer = Trainer(
        model,
        cfg,
        get_batch=lambda split: batches[split].next(),
        out_dir=args.out,
        sample_fn=lambda m: replies(m, tokenizer, PROMPTS[:3]),
    )
    trainer.train()

    text = replies(trainer.model, tokenizer, PROMPTS)
    (args.out / "replies.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
