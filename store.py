"""
Saved chats and the daily question counter.

Two stores with the same methods:
    PostgresStore   the real one (Neon PostgreSQL), used when DATABASE_URL is set
    MemoryStore     keeps everything in memory: tests, and local runs without
                    a database (chats are lost when the server stops)

Tables (created automatically on first start):

    chats        id | owner | title | created_at | updated_at
    messages     id | chat_id -> chats.id | role | content | details (JSON) | created_at
    daily_usage  owner | day | questions          (one row per person per day)

Every query includes the owner, so nobody can read or change someone
else's chats, even by guessing a chat ID.
"""

import threading
import uuid
from datetime import date, datetime, timezone
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          uuid PRIMARY KEY,
    owner       text NOT NULL,
    title       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chats_by_owner ON chats (owner, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chat_id     uuid NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    role        text NOT NULL CHECK (role IN ('user', 'assistant')),
    content     text NOT NULL,
    details     jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_by_chat ON messages (chat_id, id);

CREATE TABLE IF NOT EXISTS daily_usage (
    owner      text NOT NULL,
    day        date NOT NULL,
    questions  integer NOT NULL DEFAULT 0,
    PRIMARY KEY (owner, day)
);
"""

TITLE_CHARS = 60


def make_title(question: str) -> str:
    """A chat is named after its first question."""
    title = " ".join(question.split())
    return title if len(title) <= TITLE_CHARS else title[: TITLE_CHARS - 1] + "…"


def parse_chat_id(chat_id: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(chat_id)
    except (ValueError, TypeError, AttributeError):
        return None


class PostgresStore:
    def __init__(self, database_url: str):
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import ConnectionPool

        self._jsonb = Jsonb
        self.pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=5,
            # Neon's pooled address doesn't support prepared statements
            kwargs={"prepare_threshold": None, "row_factory": dict_row},
            # Neon sleeps when idle and drops connections: test each one
            # before use instead of failing the request
            check=ConnectionPool.check_connection,
            max_idle=300,
            open=True,
        )
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)

    def close(self) -> None:
        self.pool.close()

    def _run(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:  # commits when the block ends
            cursor = conn.execute(sql, params)
            return cursor.fetchall() if cursor.description else []

    # --- chats ---

    def list_chats(self, owner: str) -> list[dict[str, Any]]:
        return self._run(
            "SELECT id::text, title, updated_at FROM chats "
            "WHERE owner = %s ORDER BY updated_at DESC LIMIT 100",
            (owner,),
        )

    def create_chat(self, owner: str, title: str) -> str:
        chat_id = uuid.uuid4()
        self._run("INSERT INTO chats (id, owner, title) VALUES (%s, %s, %s)", (chat_id, owner, title))
        return str(chat_id)

    def chat_exists(self, owner: str, chat_id: str) -> bool:
        key = parse_chat_id(chat_id)
        return key is not None and bool(
            self._run("SELECT 1 FROM chats WHERE id = %s AND owner = %s", (key, owner))
        )

    def get_messages(self, owner: str, chat_id: str) -> list[dict[str, Any]] | None:
        """The chat's messages, oldest first; None if it isn't this person's chat."""
        if not self.chat_exists(owner, chat_id):
            return None
        return self._run(
            "SELECT role, content, details FROM messages WHERE chat_id = %s ORDER BY id",
            (parse_chat_id(chat_id),),
        )

    def add_exchange(self, owner: str, chat_id: str, question: str, answer: str, details: dict) -> None:
        """Save a question and its answer together (both or neither)."""
        key = parse_chat_id(chat_id)
        with self.pool.connection() as conn, conn.transaction():
            updated = conn.execute(
                "UPDATE chats SET updated_at = now() WHERE id = %s AND owner = %s", (key, owner)
            )
            if updated.rowcount != 1:
                raise KeyError(chat_id)
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'user', %s)",
                (key, question),
            )
            conn.execute(
                "INSERT INTO messages (chat_id, role, content, details) VALUES (%s, 'assistant', %s, %s)",
                (key, answer, self._jsonb(details)),
            )

    def delete_chat(self, owner: str, chat_id: str) -> bool:
        key = parse_chat_id(chat_id)
        if key is None:
            return False
        # messages go too (ON DELETE CASCADE)
        return bool(self._run("DELETE FROM chats WHERE id = %s AND owner = %s RETURNING id", (key, owner)))

    # --- daily question counter ---

    def questions_used(self, owner: str, day: date) -> int:
        rows = self._run("SELECT questions FROM daily_usage WHERE owner = %s AND day = %s", (owner, day))
        return rows[0]["questions"] if rows else 0

    def count_question(self, owner: str, day: date) -> int:
        # one statement, so two questions at the same moment can't both read
        # the old number and lose a count
        rows = self._run(
            "INSERT INTO daily_usage (owner, day, questions) VALUES (%s, %s, 1) "
            "ON CONFLICT (owner, day) DO UPDATE SET questions = daily_usage.questions + 1 "
            "RETURNING questions",
            (owner, day),
        )
        return rows[0]["questions"]


class MemoryStore:
    """Same methods as PostgresStore, kept in memory."""

    def __init__(self):
        self.lock = threading.Lock()
        self.chats: dict[str, dict[str, Any]] = {}
        self.usage: dict[tuple[str, date], int] = {}

    def close(self) -> None:
        pass

    def _chat(self, owner: str, chat_id: str) -> dict[str, Any] | None:
        chat = self.chats.get(str(parse_chat_id(chat_id)))
        return chat if chat and chat["owner"] == owner else None

    def list_chats(self, owner: str) -> list[dict[str, Any]]:
        mine = [c for c in self.chats.values() if c["owner"] == owner]
        mine.sort(key=lambda c: c["updated_at"], reverse=True)
        return [{"id": c["id"], "title": c["title"], "updated_at": c["updated_at"]} for c in mine[:100]]

    def create_chat(self, owner: str, title: str) -> str:
        chat_id = str(uuid.uuid4())
        with self.lock:
            self.chats[chat_id] = {
                "id": chat_id, "owner": owner, "title": title,
                "updated_at": datetime.now(timezone.utc), "messages": [],
            }
        return chat_id

    def chat_exists(self, owner: str, chat_id: str) -> bool:
        return self._chat(owner, chat_id) is not None

    def get_messages(self, owner: str, chat_id: str) -> list[dict[str, Any]] | None:
        chat = self._chat(owner, chat_id)
        return None if chat is None else [dict(m) for m in chat["messages"]]

    def add_exchange(self, owner: str, chat_id: str, question: str, answer: str, details: dict) -> None:
        with self.lock:
            chat = self._chat(owner, chat_id)
            if chat is None:
                raise KeyError(chat_id)
            chat["messages"].append({"role": "user", "content": question, "details": None})
            chat["messages"].append({"role": "assistant", "content": answer, "details": details})
            chat["updated_at"] = datetime.now(timezone.utc)

    def delete_chat(self, owner: str, chat_id: str) -> bool:
        with self.lock:
            if self._chat(owner, chat_id) is None:
                return False
            del self.chats[str(parse_chat_id(chat_id))]
            return True

    def questions_used(self, owner: str, day: date) -> int:
        return self.usage.get((owner, day), 0)

    def count_question(self, owner: str, day: date) -> int:
        with self.lock:
            self.usage[(owner, day)] = self.usage.get((owner, day), 0) + 1
            return self.usage[(owner, day)]


def make_store(database_url: str | None) -> PostgresStore | MemoryStore:
    return PostgresStore(database_url) if database_url else MemoryStore()
