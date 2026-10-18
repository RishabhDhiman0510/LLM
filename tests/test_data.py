import numpy as np
import torch

from wikigpt.data import NUM_SHARDS, TokenDataset, append_tokens, clean_article, encode_articles, shard_order


def test_clean_article_cuts_tail_sections():
    body = "Paris is the capital of France. " * 30
    text = body + "\n\nSee also\n\nLyon\n\nReferences\n"
    assert clean_article(text) == body.strip()


def test_clean_article_drops_stubs():
    assert clean_article("A short stub.") is None


def test_clean_article_collapses_blank_lines():
    text = "a" * 300 + "\n\n\n\n" + "b" * 300
    assert clean_article(text) == "a" * 300 + "\n\n" + "b" * 300


def test_shard_order_is_a_seeded_permutation():
    assert shard_order(7) == shard_order(7)
    assert sorted(shard_order(7)) == list(range(NUM_SHARDS))


def test_encode_articles_ends_each_with_eot(tokenizer):
    ids = encode_articles(tokenizer, ["Hello there.", "Paris is a city."])
    assert ids.dtype == np.uint16
    assert (ids == tokenizer.eot_id).sum() == 2
    assert ids[-1] == tokenizer.eot_id


def test_token_dataset_targets_are_shifted_by_one(tmp_path):
    path = tmp_path / "train.bin"
    append_tokens(path, np.arange(100))
    append_tokens(path, np.arange(100, 200))
    dataset = TokenDataset(path, context_length=8, seed=0)
    assert dataset.size == 200
    x, y = dataset.batch(4)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)
