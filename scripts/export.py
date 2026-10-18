import argparse
import copy
import shutil
from pathlib import Path

import torch

from wikigpt.chat import chat_reply
from wikigpt.checkpoint import export_bundle, load_bundle, load_checkpoint, model_from_checkpoint
from wikigpt.model import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a chat-tuned checkpoint as the bundle the local app loads.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="finetune ckpt.pt with tokenizer.json next to it")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/export"))
    parser.add_argument("--name", default="wikigpt-12m")
    return parser.parse_args()


def verify(model: GPT, bundle: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded, tokenizer = load_bundle(bundle, device)
    # the bundle is fp16, so compare against the same weights rounded to fp16
    reference = copy.deepcopy(model).half().float().to(device)
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    expected = chat_reply(reference, tokenizer, messages, temperature=0.0)
    actual = chat_reply(loaded, tokenizer, messages, temperature=0.0)
    if expected != actual:
        raise SystemExit("export check failed: the reloaded bundle does not match the checkpoint")
    print(f"export check passed, sample reply: {actual}")


def main() -> None:
    args = parse_args()
    model = model_from_checkpoint(load_checkpoint(args.checkpoint), dropout=0.0).eval()
    bundle = export_bundle(model, args.checkpoint.parent / "tokenizer.json", args.out / args.name)
    verify(model, bundle)
    archive = Path(shutil.make_archive(str(args.out / args.name), "zip", root_dir=args.out, base_dir=args.name))
    for path in sorted(bundle.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
    print(f"{archive.name}: {archive.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
