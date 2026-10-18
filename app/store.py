import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TITLE_LENGTH = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def make_title(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= TITLE_LENGTH else line[: TITLE_LENGTH - 3].rstrip() + "..."


class ChatStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def create_chat(self) -> dict:
        stamp = now()
        with self.connect() as db:
            cursor = db.execute("INSERT INTO chats (created_at, updated_at) VALUES (?, ?)", (stamp, stamp))
            return self._fetch_chat(db, cursor.lastrowid)

    def get_chat(self, chat_id: int) -> dict | None:
        with self.connect() as db:
            return self._fetch_chat(db, chat_id)

    def list_chats(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM chats ORDER BY updated_at DESC, id DESC").fetchall()
        return [dict(row) for row in rows]

    def delete_chat(self, chat_id: int) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def get_messages(self, chat_id: int) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, chat_id: int, role: str, content: str) -> dict:
        stamp = now()
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, stamp),
            )
            db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (stamp, chat_id))
            if role == "user":
                db.execute("UPDATE chats SET title = ? WHERE id = ? AND title = ''", (make_title(content), chat_id))
            row = db.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    @staticmethod
    def _fetch_chat(db: sqlite3.Connection, chat_id: int) -> dict | None:
        row = db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return dict(row) if row else None
