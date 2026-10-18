import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .chat import IGNORE_INDEX, Message, build_example
from .tokenizer import Tokenizer

ALPACA_REPO = "yahma/alpaca-cleaned"
EVERYDAY_REPO = "HuggingFaceTB/everyday-conversations-llama3.1-2k"

Conversation = list[Message]
Example = tuple[list[int], list[int]]


def load_alpaca() -> list[Conversation]:
    path = hf_hub_download(ALPACA_REPO, "alpaca_data_cleaned.json", repo_type="dataset")
    conversations = []
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        prompt = row["instruction"].strip()
        if row["input"].strip():
            prompt += "\n\n" + row["input"].strip()
        if row["output"].strip():
            conversations.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"]}])
    return conversations


def load_everyday() -> list[Conversation]:
    rows = load_dataset(EVERYDAY_REPO, split="train_sft")
    return [[m for m in row["messages"] if m["role"] in ("user", "assistant")] for row in rows]


def split(items: list, fraction: float, rng: random.Random) -> tuple[list, list]:
    items = items[:]
    rng.shuffle(items)
    cut = max(1, int(len(items) * fraction))
    return items[cut:], items[:cut]


def encode_all(tokenizer: Tokenizer, conversations: list[Conversation], max_len: int) -> list[Example]:
    examples = (build_example(tokenizer, conversation, max_len) for conversation in conversations)
    return [example for example in examples if example is not None]


def build_sft_data(
    tokenizer: Tokenizer,
    context_length: int,
    seed: int,
    val_fraction: float = 0.02,
    everyday_repeat: int = 3,
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    alpaca_train, alpaca_val = split(load_alpaca(), val_fraction, rng)
    everyday_train, everyday_val = split(load_everyday(), val_fraction, rng)
    # the small-talk set is tiny, so it is repeated to keep greetings in the mix
    train = alpaca_train + everyday_train * everyday_repeat
    val = alpaca_val + everyday_val
    max_len = context_length + 1
    return encode_all(tokenizer, train, max_len), encode_all(tokenizer, val, max_len)


def collate(batch: list[Example], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids, _ in batch) - 1
    x = torch.full((len(batch), width), pad_id, dtype=torch.long)
    y = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, labels) in enumerate(batch):
        n = len(ids) - 1
        x[row, :n] = torch.tensor(ids[:-1])
        y[row, :n] = torch.tensor(labels[1:])
    return x, y


class SFTBatches:
    def __init__(self, examples: list[Example], batch_size: int, pad_id: int, seed: int):
        self.examples = examples
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.rng = random.Random(seed)
        self.order: list[int] = []

    def __len__(self) -> int:
        return max(1, len(self.examples) // self.batch_size)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.order) < self.batch_size:
            self.order = list(range(len(self.examples)))
            self.rng.shuffle(self.order)
        picked, self.order = self.order[: self.batch_size], self.order[self.batch_size :]
        return collate([self.examples[i] for i in picked], self.pad_id)
