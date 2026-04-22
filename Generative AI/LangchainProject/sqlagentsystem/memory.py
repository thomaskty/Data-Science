from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


@dataclass
class MemoryTurn:
    user_text: str
    agent_text: str
    created_at: str


class MemoryManager:
    """Short-term + lightweight long-term memory for chat-style agents."""

    def __init__(self, max_messages: int, memory_file: Path) -> None:
        self._max_messages = max_messages
        self._messages: deque[BaseMessage] = deque(maxlen=max_messages)
        self._memory_file = memory_file
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)

        self._vector_store: InMemoryVectorStore | None = None
        self._init_vector_store()
        self._load_history()

    def _init_vector_store(self) -> None:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._vector_store = InMemoryVectorStore(embedding=embeddings)
        except Exception:
            # Embeddings are optional; short-term memory still works.
            self._vector_store = None

    def _load_history(self) -> None:
        if not self._memory_file.exists():
            return

        try:
            lines = self._memory_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            return

        records: list[MemoryTurn] = []
        for raw in lines:
            try:
                payload = json.loads(raw)
                user_text = str(payload["user_text"])
                agent_text = str(payload["agent_text"])
                created_at = str(payload.get("created_at", ""))
                records.append(
                    MemoryTurn(
                        user_text=user_text,
                        agent_text=agent_text,
                        created_at=created_at,
                    )
                )
            except Exception:
                continue

        for turn in records[-(self._max_messages // 2) :]:
            self._messages.append(HumanMessage(content=turn.user_text))
            self._messages.append(AIMessage(content=turn.agent_text))

        if self._vector_store and records:
            texts = [self._to_vector_text(r.user_text, r.agent_text) for r in records]
            metadatas = [
                {"created_at": r.created_at, "source": "chat_history"} for r in records
            ]
            ids = [str(uuid.uuid4()) for _ in records]
            try:
                self._vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            except Exception:
                self._vector_store = None

    def short_term_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def add_turn(self, user_text: str, agent_text: str) -> None:
        self._messages.append(HumanMessage(content=user_text))
        self._messages.append(AIMessage(content=agent_text))

        record = MemoryTurn(
            user_text=user_text,
            agent_text=agent_text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append_to_disk(record)
        if self._vector_store:
            try:
                self._vector_store.add_texts(
                    texts=[self._to_vector_text(user_text, agent_text)],
                    metadatas=[
                        {"created_at": record.created_at, "source": "chat_history"}
                    ],
                    ids=[str(uuid.uuid4())],
                )
            except Exception:
                self._vector_store = None

    def recall(self, query: str, k: int = 4) -> list[str]:
        if not self._vector_store:
            return []
        try:
            docs = self._vector_store.similarity_search(query=query, k=k)
            return [str(doc.page_content) for doc in docs]
        except Exception:
            return []

    def _append_to_disk(self, turn: MemoryTurn) -> None:
        payload = {
            "user_text": turn.user_text,
            "agent_text": turn.agent_text,
            "created_at": turn.created_at,
        }
        with self._memory_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    @staticmethod
    def _to_vector_text(user_text: str, agent_text: str) -> str:
        return f"User: {user_text}\nAgent: {agent_text}"
