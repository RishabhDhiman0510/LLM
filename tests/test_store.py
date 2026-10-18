from app.store import ChatStore


def test_chat_lifecycle(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    chat = store.create_chat()
    assert chat["title"] == ""

    store.add_message(chat["id"], "user", "What is the tallest mountain on Earth and how tall is it?")
    store.add_message(chat["id"], "assistant", "Mount Everest.")
    store.add_message(chat["id"], "user", "And the second tallest?")

    messages = store.get_messages(chat["id"])
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    title = store.get_chat(chat["id"])["title"]
    assert title.startswith("What is the tallest mountain")
    assert len(title) <= 40

    store.delete_chat(chat["id"])
    assert store.get_chat(chat["id"]) is None
    assert store.get_messages(chat["id"]) == []


def test_most_recently_used_chat_is_listed_first(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    first = store.create_chat()
    second = store.create_chat()
    store.add_message(first["id"], "user", "Hello")
    assert [c["id"] for c in store.list_chats()] == [first["id"], second["id"]]
