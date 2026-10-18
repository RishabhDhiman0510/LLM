import random
import re
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from .tokenizer import Tokenizer

WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"
NUM_SHARDS = 41
MIN_CHARS = 500

TAIL_SECTION = re.compile(r"\n(?:See also|References|Notes|Further reading|External links|Bibliography|Sources)[ \t]*\n")
BLANK_LINES = re.compile(r"\n{3,}")


def shard_order(seed: int) -> list[int]:
    order = list(range(NUM_SHARDS))
    random.Random(seed).shuffle(order)
    return order


def download_shard(index: int, cache_dir: Path, retries: int = 3) -> Path:
    filename = f"{WIKI_CONFIG}/train-{index:05d}-of-{NUM_SHARDS:05d}.parquet"
    for attempt in range(1, retries + 1):
        try:
            return Path(hf_hub_download(WIKI_REPO, filename, repo_type="dataset", local_dir=cache_dir))
        except Exception as error:
            if attempt == retries:
                raise
            print(f"download failed ({error}), retrying in {30 * attempt}s")
            time.sleep(30 * attempt)
    raise RuntimeError("unreachable")


def clean_article(text: str, min_chars: int = MIN_CHARS) -> str | None:
    tail = TAIL_SECTION.search(text)
    if tail:
        text = text[: tail.start()]
    text = BLANK_LINES.sub("\n\n", text).strip()
    return text if len(text) >= min_chars else None


def read_articles(path: Path, batch_size: int) -> Iterator[list[str]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["text"]):
        cleaned = (clean_article(text) for text in batch.column(0).to_pylist())
        yield [text for text in cleaned if text]


def encode_articles(tokenizer: Tokenizer, articles: list[str]) -> np.ndarray:
    if not articles:
        return np.empty(0, dtype=np.uint16)
    return np.concatenate([np.array(ids + [tokenizer.eot_id], dtype=np.uint16) for ids in tokenizer.encode_batch(articles)])


def append_tokens(path: Path, ids: np.ndarray) -> None:
    with path.open("ab") as f:
        ids.astype(np.uint16).tofile(f)


class TokenDataset:
    def __init__(self, path: Path, context_length: int, seed: int = 0):
        self.path = path
        self.context_length = context_length
        self.size = path.stat().st_size // 2
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # reopen the memmap each call so pages are not held across batches
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        starts = self.rng.integers(0, self.size - self.context_length - 1, size=batch_size)
        windows = np.stack([data[s : s + self.context_length + 1] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(windows)
        return tokens[:, :-1], tokens[:, 1:]
