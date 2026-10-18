from .generate import generate
from .model import GPT
from .tokenizer import Tokenizer

IGNORE_INDEX = -100

Message = dict[str, str]


def encode_turn(tokenizer: Tokenizer, role: str, text: str) -> list[int]:
    role_id = tokenizer.user_id if role == "user" else tokenizer.assistant_id
    # typed markers such as <|end|> must stay plain text, not control tokens
    text = text.strip().replace("<|", "< |")
    return [role_id, *tokenizer.encode("\n"), *tokenizer.encode(text), tokenizer.end_id]


def reply_header(tokenizer: Tokenizer) -> list[int]:
    return [tokenizer.assistant_id, *tokenizer.encode("\n")]


def build_example(tokenizer: Tokenizer, messages: list[Message], max_len: int) -> tuple[list[int], list[int]] | None:
    header = len(reply_header(tokenizer))
    ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        turn = encode_turn(tokenizer, message["role"], message["content"])
        ids += turn
        if message["role"] == "assistant":
            # the header is supplied at inference time, so only the reply is learned
            labels += [IGNORE_INDEX] * header + turn[header:]
        else:
            labels += [IGNORE_INDEX] * len(turn)
    if len(ids) > max_len or all(label == IGNORE_INDEX for label in labels):
        return None
    return ids, labels


def build_prompt(tokenizer: Tokenizer, messages: list[Message], max_tokens: int) -> list[int]:
    header = reply_header(tokenizer)
    budget = max_tokens - len(header)
    if budget <= 2 * len(header):
        raise ValueError(f"max_tokens={max_tokens} leaves no room for the conversation")
    kept: list[int] = []
    for message in reversed(messages):
        turn = encode_turn(tokenizer, message["role"], message["content"])
        if len(kept) + len(turn) > budget:
            if not kept:
                # an over-long latest message keeps its role marker and its last tokens
                kept = turn[: len(header)] + turn[len(header) - budget :]
            break
        kept = turn + kept
    return kept + header


def chat_reply(
    model: GPT,
    tokenizer: Tokenizer,
    messages: list[Message],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
) -> str:
    prompt = build_prompt(tokenizer, messages, model.cfg.context_length - max_new_tokens)
    stop = {tokenizer.end_id, tokenizer.user_id, tokenizer.eot_id}
    ids = generate(model, prompt, max_new_tokens, temperature, top_k, repetition_penalty, stop)
    return tokenizer.decode(ids).strip()
