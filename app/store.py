"""SQLite storage for documents, indexed text and chunks, presets and run history."""

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS indexed_documents (
    index_key TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (index_key, document_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    index_key TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_index ON chunks (index_key, document_id, idx);
CREATE TABLE IF NOT EXISTS presets (
    name TEXT PRIMARY KEY,
    overrides_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock, self._db:
            return self._db.execute(sql, params).fetchall()

    # Documents

    def add_document(self, doc_id: str, filename: str, path: Path, size_bytes: int) -> dict[str, Any]:
        self._execute(
            "INSERT INTO documents (id, filename, path, size_bytes, created_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, str(path), size_bytes, now()),
        )
        return self.get_document(doc_id)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        rows = self._execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return self._with_indexes(dict(rows[0])) if rows else None

    def list_documents(self) -> list[dict[str, Any]]:
        return [self._with_indexes(dict(r)) for r in self._execute("SELECT * FROM documents ORDER BY created_at DESC")]

    def _with_indexes(self, doc: dict[str, Any]) -> dict[str, Any]:
        rows = self._execute(
            "SELECT index_key, chunk_count, created_at FROM indexed_documents WHERE document_id = ? ORDER BY created_at",
            (doc["id"],),
        )
        doc["indexes"] = [dict(r) for r in rows]
        return doc

    def delete_document(self, doc_id: str) -> None:
        self._execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # Indexed text and chunks

    def save_index(self, index_key: str, doc_id: str, text: str, chunks: list[Chunk], config: dict) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE index_key = ? AND document_id = ?", (index_key, doc_id))
            self._db.execute(
                "INSERT OR REPLACE INTO indexed_documents VALUES (?, ?, ?, ?, ?, ?)",
                (index_key, doc_id, text, len(chunks), json.dumps(config), now()),
            )
            self._db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, index_key, doc_id, c.index, c.text, c.start_char, c.end_char, json.dumps(c.metadata))
                    for c in chunks
                ],
            )

    def is_indexed(self, index_key: str, doc_id: str) -> bool:
        return bool(
            self._execute("SELECT 1 FROM indexed_documents WHERE index_key = ? AND document_id = ?", (index_key, doc_id))
        )

    def get_indexed_text(self, index_key: str, doc_id: str) -> str | None:
        rows = self._execute(
            "SELECT text FROM indexed_documents WHERE index_key = ? AND document_id = ?", (index_key, doc_id)
        )
        return rows[0]["text"] if rows else None

    def get_chunks(self, index_key: str, doc_ids: list[str]) -> list[Chunk]:
        placeholders = ",".join("?" * len(doc_ids))
        rows = self._execute(
            f"SELECT * FROM chunks WHERE index_key = ? AND document_id IN ({placeholders}) ORDER BY document_id, idx",
            (index_key, *doc_ids),
        )
        return [
            Chunk(
                id=r["id"],
                document_id=r["document_id"],
                index=r["idx"],
                text=r["text"],
                start_char=r["start_char"],
                end_char=r["end_char"],
                metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]

    # Presets

    def list_presets(self) -> list[dict[str, Any]]:
        rows = self._execute("SELECT * FROM presets ORDER BY name")
        return [{"name": r["name"], "overrides": json.loads(r["overrides_json"]), "updated_at": r["updated_at"]} for r in rows]

    def get_preset(self, name: str) -> dict[str, Any] | None:
        return next((p for p in self.list_presets() if p["name"] == name), None)

    def save_preset(self, name: str, overrides: dict) -> dict[str, Any]:
        self._execute("INSERT OR REPLACE INTO presets VALUES (?, ?, ?)", (name, json.dumps(overrides), now()))
        return self.get_preset(name)

    def delete_preset(self, name: str) -> bool:
        exists = self.get_preset(name) is not None
        self._execute("DELETE FROM presets WHERE name = ?", (name,))
        return exists

    # Runs

    def save_run(self, run_id: str, question: str, result: dict) -> None:
        self._execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (run_id, question, json.dumps(result), now()))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        return {**json.loads(rows[0]["result_json"]), "created_at": rows[0]["created_at"]} if rows else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._execute("SELECT id, question, created_at FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
