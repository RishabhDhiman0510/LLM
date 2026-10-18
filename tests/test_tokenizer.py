from wikigpt.tokenizer import ASSISTANT, END, EOT, USER, Tokenizer


def test_roundtrip_is_exact(tokenizer):
    text = "Paris is the capital of France.\n\nIt had 2,102,650 residents in 2023, cafes included."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_handles_unicode(tokenizer):
    text = "Zoë visited São Paulo and Kraków."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_single_ids(tokenizer):
    assert tokenizer.encode(EOT) == [tokenizer.eot_id]
    assert tokenizer.encode(USER) == [tokenizer.user_id]
    assert tokenizer.encode(ASSISTANT) == [tokenizer.assistant_id]
    assert tokenizer.encode(END) == [tokenizer.end_id]


def test_save_and_load(tokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    text = "The quick brown fox."
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.end_id == tokenizer.end_id
