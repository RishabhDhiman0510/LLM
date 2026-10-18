import torch

from wikigpt.config import ModelConfig
from wikigpt.generate import generate
from wikigpt.model import GPT

TINY = ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4)


def test_forward_shapes():
    model = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (2, TINY.context_length))
    logits, loss = model(x, x)
    assert logits.shape == (2, TINY.context_length, TINY.vocab_size)
    assert loss.ndim == 0


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, TINY.context_length))
    changed = x.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % TINY.vocab_size
    before, _ = model(x)
    after, _ = model(changed)
    assert torch.allclose(before[0, :10], after[0, :10], atol=1e-5)
    assert not torch.allclose(before[0, 10:], after[0, 10:])


def test_weights_are_tied():
    model = GPT(TINY)
    assert model.head.weight is model.tok_emb.weight


def test_default_model_is_about_12m_params():
    assert GPT(ModelConfig()).num_params() == 12_649_600


def test_loss_falls_when_overfitting_one_batch():
    torch.manual_seed(0)
    model = GPT(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, TINY.vocab_size, (4, TINY.context_length + 1))
    losses = []
    for _ in range(30):
        _, loss = model(x[:, :-1], x[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 1.0


def test_generate_slides_past_the_context_window():
    torch.manual_seed(0)
    model = GPT(TINY)
    out = generate(model, [1, 2, 3], max_new_tokens=20, repetition_penalty=1.3, seed=0)
    assert len(out) == 20
    assert all(0 <= token < TINY.vocab_size for token in out)


def test_generate_stops_on_stop_id():
    torch.manual_seed(0)
    model = GPT(TINY)
    first = generate(model, [1, 2, 3], max_new_tokens=1, temperature=0.0)[0]
    assert generate(model, [1, 2, 3], max_new_tokens=5, temperature=0.0, stop_ids={first}) == []


def test_ignored_targets_do_not_count():
    model = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (1, 8))
    targets = torch.full((1, 8), -100)
    targets[0, -1] = 3
    _, loss = model(x, targets)
    assert torch.isfinite(loss)
