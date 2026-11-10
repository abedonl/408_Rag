"""错题本存储：匿名 client_id + SQLite。

明确取舍：不引入账号系统。前端生成匿名 client_id（写进 URL 查询串，
刷新/收藏不丢），后端按 (client_id, qid) 存错题。判分逻辑在 judge.py。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cs408rag.config import DATA_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wrong_items (
    client_id      TEXT NOT NULL,
    qid            TEXT NOT NULL,
    subject        TEXT NOT NULL DEFAULT '',
    tags           TEXT NOT NULL DEFAULT '[]',
    your_choice    TEXT NOT NULL DEFAULT '',
    correct_choice TEXT NOT NULL DEFAULT '',
    wrong_count    INTEGER NOT NULL DEFAULT 1,
    first_wrong_at TEXT NOT NULL,
    last_wrong_at  TEXT NOT NULL,
    PRIMARY KEY (client_id, qid)
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    return d


class NotebookStore:
    """单连接 + 锁：错题本是低频单写场景，比每次新建连接简单。"""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (DATA_DIR / "notebook.db")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def record_wrong(
        self,
        client_id: str,
        qid: str,
        *,
        your_choice: str,
        correct_choice: str,
        subject: str,
        tags: list[str],
    ) -> dict[str, Any]:
        """答错落库：已存在则 wrong_count+1 并更新最近作答。"""
        now = _now()
        tags_json = json.dumps(tags, ensure_ascii=False)
        with self._lock:
            row = self._conn.execute(
                "SELECT wrong_count FROM wrong_items WHERE client_id=? AND qid=?",
                (client_id, qid),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE wrong_items SET your_choice=?, correct_choice=?, subject=?, "
                    "tags=?, wrong_count=?, last_wrong_at=? WHERE client_id=? AND qid=?",
                    (
                        your_choice, correct_choice, subject, tags_json,
                        row["wrong_count"] + 1, now, client_id, qid,
                    ),
                )
            else:
                self._conn.execute(
                    "INSERT INTO wrong_items (client_id, qid, subject, tags, "
                    "your_choice, correct_choice, wrong_count, first_wrong_at, "
                    "last_wrong_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        client_id, qid, subject, tags_json,
                        your_choice, correct_choice, 1, now, now,
                    ),
                )
            self._conn.commit()
        item = self.get(client_id, qid)
        assert item is not None  # 刚写入，必然存在
        return item

    def remove(self, client_id: str, qid: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM wrong_items WHERE client_id=? AND qid=?", (client_id, qid)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get(self, client_id: str, qid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM wrong_items WHERE client_id=? AND qid=?",
                (client_id, qid),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_items(self, client_id: str, subject: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM wrong_items WHERE client_id=?"
        args: list[str] = [client_id]
        if subject:
            sql += " AND subject=?"
            args.append(subject)
        sql += " ORDER BY last_wrong_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]


_store: NotebookStore | None = None
_store_lock = threading.Lock()


def get_store() -> NotebookStore:
    """进程级单例（参照 agent/memory.py 的 get_session_store）。"""
    global _store
    with _store_lock:
        if _store is None:
            _store = NotebookStore()
    return _store
