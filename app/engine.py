import threading

import torch

from wikigpt.chat import chat_reply
from wikigpt.checkpoint import load_bundle

from .settings import Settings


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("WIKIGPT_DEVICE is cuda, but this torch install cannot see a GPU")
    return name


class ChatEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.model, self.tokenizer = load_bundle(settings.model_dir, self.device)
        self.lock = threading.Lock()
        limit = self.model.cfg.context_length // 2
        if not 0 < settings.max_new_tokens <= limit:
            raise ValueError(f"WIKIGPT_MAX_NEW_TOKENS must be between 1 and {limit}")

    def reply(self, messages: list[dict]) -> str:
        s = self.settings
        with self.lock:
            return chat_reply(
                self.model,
                self.tokenizer,
                messages,
                max_new_tokens=s.max_new_tokens,
                temperature=s.temperature,
                top_k=s.top_k,
                repetition_penalty=s.repetition_penalty,
            )
