from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
SPECIAL_TOKENS = [EOT, USER, ASSISTANT, END]


class Tokenizer:
    def __init__(self, backend: HFTokenizer):
        self.backend = backend
        self.eot_id = backend.token_to_id(EOT)
        self.user_id = backend.token_to_id(USER)
        self.assistant_id = backend.token_to_id(ASSISTANT)
        self.end_id = backend.token_to_id(END)

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [encoding.ids for encoding in self.backend.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode(ids, skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        self.backend.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(HFTokenizer.from_file(str(path)))


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    backend = HFTokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(texts, trainer=trainer)
    return Tokenizer(backend)
