from collections.abc import Collection

import torch

from .model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.0,
    stop_ids: Collection[int] = (),
    seed: int | None = None,
) -> list[int]:
    model.eval()
    device = next(model.parameters()).device
    context = model.cfg.context_length
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_new_tokens):
        window = x[:, -context:]
        logits = model(window)[0][0, -1].float()
        # penalise only tokens this reply already used, so template tokens like <|end|> are left alone
        if repetition_penalty != 1.0 and out:
            seen = torch.tensor(sorted(set(out)), device=device)
            scores = logits[seen]
            logits[seen] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
        if temperature <= 0:
            next_id = logits.argmax().view(1)
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
        token = int(next_id)
        if token in stop_ids:
            break
        out.append(token)
        x = torch.cat([x, next_id.view(1, 1)], dim=1)
    return out
