import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Wikipedia is a free online encyclopedia written by volunteers.",
    "Paris is the capital and largest city of France.",
    "In mathematics, a prime number has exactly two divisors.",
    "Hello, how are you today? I am fine, thank you.",
]


@pytest.fixture(scope="session")
def tokenizer():
    from wikigpt.tokenizer import train_tokenizer

    return train_tokenizer(CORPUS * 20, vocab_size=320)
