"""The chat store: the same tests run against MemoryStore and real PostgreSQL.

PostgreSQL tests run when TEST_DATABASE_URL points at a throwaway database
(CI starts one; locally: docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:18).
"""

import os
import uuid
from datetime import date
from types import SimpleNamespace

import pytest

import store as chat_store

DAY = date(2026, 11, 13)


@pytest.fixture(params=["memory", "postgres"])
def store(request):
    if request.param == "memory":
        yield chat_store.MemoryStore()
        return
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    db = chat_store.PostgresStore(url)
    yield db
    db.close()


@pytest.fixture
def people():
    # unique emails per test, so tests never see each other's rows
    tag = uuid.uuid4().hex[:8]
    return SimpleNamespace(me=f"me-{tag}@test.com", other=f"other-{tag}@test.com")


def test_a_chat_keeps_its_messages_in_order(store, people):
    chat = store.create_chat(people.me, "Loading filament")
    store.add_exchange(people.me, chat, "How do I load filament?", "Push it in.", {"covered": True})
    store.add_exchange(people.me, chat, "And unload?", "Pull it out.", {"covered": True})

    assert store.get_messages(people.me, chat) == [
        {"role": "user", "content": "How do I load filament?", "details": None},
        {"role": "assistant", "content": "Push it in.", "details": {"covered": True}},
        {"role": "user", "content": "And unload?", "details": None},
        {"role": "assistant", "content": "Pull it out.", "details": {"covered": True}},
    ]


def test_newest_chat_first(store, people):
    first = store.create_chat(people.me, "first")
    second = store.create_chat(people.me, "second")
    store.add_exchange(people.me, first, "q", "a", {})  # first is now the latest
    assert [c["title"] for c in store.list_chats(people.me)] == ["first", "second"]
    assert {c["id"] for c in store.list_chats(people.me)} == {first, second}


def test_nobody_can_touch_someone_elses_chat(store, people):
    chat = store.create_chat(people.me, "private")

    assert store.list_chats(people.other) == []
    assert store.get_messages(people.other, chat) is None
    assert store.chat_exists(people.other, chat) is False
    assert store.delete_chat(people.other, chat) is False
    with pytest.raises(KeyError):
        store.add_exchange(people.other, chat, "q", "a", {})
    assert store.get_messages(people.me, chat) == []  # untouched


def test_delete_removes_the_chat_and_its_messages(store, people):
    chat = store.create_chat(people.me, "temp")
    store.add_exchange(people.me, chat, "q", "a", {})
    assert store.delete_chat(people.me, chat) is True
    assert store.get_messages(people.me, chat) is None
    assert store.list_chats(people.me) == []


def test_made_up_chat_ids_are_just_not_found(store, people):
    for bad in ["not-a-uuid", "", str(uuid.uuid4())]:
        assert store.get_messages(people.me, bad) is None
        assert store.delete_chat(people.me, bad) is False


def test_daily_counter(store, people):
    assert store.questions_used(people.me, DAY) == 0
    assert store.count_question(people.me, DAY) == 1
    assert store.count_question(people.me, DAY) == 2
    assert store.questions_used(people.me, DAY) == 2
    assert store.questions_used(people.other, DAY) == 0     # per person
    assert store.questions_used(people.me, date(2026, 11, 14)) == 0  # per day


def test_titles_are_short_and_on_one_line():
    assert chat_store.make_title("  How do I\\n load   filament? ".replace("\\n", "\n")) == "How do I load filament?"
    long = chat_store.make_title("word " * 40)
    assert len(long) == chat_store.TITLE_CHARS and long.endswith("…")
