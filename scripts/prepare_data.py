import argparse
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

from wikigpt.data import WIKI_CONFIG, WIKI_REPO, append_tokens, download_shard, encode_articles, read_articles, shard_order
from wikigpt.tokenizer import Tokenizer, train_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Wikipedia shards, train the tokenizer and write token files.")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/data"))
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-chars", type=int, default=300_000_000)
    parser.add_argument("--encode-batch", type=int, default=512)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.vocab_size > 65535:
        parser.error("--vocab-size must fit in uint16 (65535 or less)")
    return args


def load_meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"source": f"{WIKI_REPO}/{WIKI_CONFIG}", "shards": [], "articles": 0, "chars": 0, "train_tokens": 0, "val_tokens": 0}


def truncate(path: Path, tokens: int) -> None:
    # drop anything written by a shard that did not finish
    if path.exists():
        os.truncate(path, tokens * 2)
    else:
        path.touch()


def tokenizer_texts(path: Path, max_chars: int, batch_size: int) -> Iterator[str]:
    total = 0
    for articles in read_articles(path, batch_size):
        for text in articles:
            yield text
            total += len(text)
            if total >= max_chars:
                return


def encode_shard(path: Path, tokenizer: Tokenizer, meta: dict, out: Path, args: argparse.Namespace) -> None:
    for articles in read_articles(path, args.encode_batch):
        train, val = [], []
        for text in articles:
            (val if meta["articles"] % args.val_every == 0 else train).append(text)
            meta["articles"] += 1
            meta["chars"] += len(text)
        train_ids = encode_articles(tokenizer, train)
        val_ids = encode_articles(tokenizer, val)
        append_tokens(out / "train.bin", train_ids)
        append_tokens(out / "val.bin", val_ids)
        meta["train_tokens"] += len(train_ids)
        meta["val_tokens"] += len(val_ids)
        if meta["train_tokens"] >= args.target_tokens:
            return


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    tokenizer_path = out / "tokenizer.json"
    cache_dir = out / "shards"

    meta = load_meta(meta_path)
    truncate(out / "train.bin", meta["train_tokens"])
    truncate(out / "val.bin", meta["val_tokens"])
    tokenizer = Tokenizer.load(tokenizer_path) if tokenizer_path.exists() else None
    start = time.time()

    for shard in shard_order(args.seed):
        if meta["train_tokens"] >= args.target_tokens:
            break
        if shard in meta["shards"]:
            continue
        print(f"shard {shard}: downloading", flush=True)
        path = download_shard(shard, cache_dir)
        if tokenizer is None:
            print(f"training tokenizer on {args.tokenizer_chars / 1e6:.0f}M characters", flush=True)
            tokenizer = train_tokenizer(tokenizer_texts(path, args.tokenizer_chars, args.encode_batch), args.vocab_size)
            tokenizer.save(tokenizer_path)
        print(f"shard {shard}: encoding", flush=True)
        encode_shard(path, tokenizer, meta, out, args)
        path.unlink()
        meta["shards"].append(shard)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"shard {shard}: done, {meta['train_tokens'] / 1e6:.1f}M train tokens, {(time.time() - start) / 60:.1f} min", flush=True)

    shutil.rmtree(cache_dir, ignore_errors=True)
    meta["vocab_size"] = tokenizer.vocab_size
    meta["chars_per_token"] = round(meta["chars"] / (meta["train_tokens"] + meta["val_tokens"]), 3)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
