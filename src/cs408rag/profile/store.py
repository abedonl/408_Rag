"""画像事件存储：匿名 client_id + SQLite，模式同 notebook/store.py。"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cs408rag.config import DATA_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    kind      TEXT NOT NULL,
    payload   TEXT NOT NULL DEFAULT '{}',
    ts        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_events_cid ON profile_events(client_id, id);
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_cid ON conversations(client_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '{}',
    ts              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id    TEXT NOT NULL,
    period_type  TEXT NOT NULL,
    period_date  TEXT NOT NULL,
    content      TEXT NOT NULL DEFAULT '',
    generated_at TEXT NOT NULL,
    UNIQUE(client_id, period_type, period_date)
);
CREATE INDEX IF NOT EXISTS idx_summaries_cid ON summaries(client_id, period_type, period_date);
CREATE TABLE IF NOT EXISTS plans (
    client_id  TEXT PRIMARY KEY,
    plan_json  TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ProfileStore:
    """单连接 + 锁：画像事件是低频单写场景，比每次新建连接简单。"""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (DATA_DIR / "profile.db")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record(self, client_id: str, kind: str, payload: dict[str, Any]) -> None:
        """追加一条事件。kind: question / plan / quiz。"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO profile_events (client_id, kind, payload, ts) VALUES (?,?,?,?)",
                (client_id, kind, json.dumps(payload, ensure_ascii=False), _now()),
            )
            self._conn.commit()

    def events(
        self, client_id: str, kind: str | None = None, *, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """按时间正序返回事件，payload 已解析为 dict。"""
        sql = "SELECT kind, payload, ts FROM profile_events WHERE client_id=?"
        args: list[Any] = [client_id]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = [
            {"kind": r["kind"], "payload": json.loads(r["payload"] or "{}"), "ts": r["ts"]}
            for r in rows
        ]
        out.reverse()  # 取最近 limit 条后恢复时间正序
        return out

    def clear(self, client_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM profile_events WHERE client_id=?", (client_id,)
            )
            self._conn.commit()
            return cur.rowcount

    # -- 会话持久化 -----------------------------------------------------

    def list_conversations(self, client_id: str) -> list[dict[str, Any]]:
        """该用户的会话，按更新时间倒序。

        n_messages 供侧栏过滤空会话用（空会话点进去没内容，是噪声）。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS n "
                "FROM conversations c WHERE c.client_id=? ORDER BY c.updated_at DESC",
                (client_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "n_messages": r["n"],
            }
            for r in rows
        ]

    def create_conversation(self, client_id: str, conv_id: str) -> dict[str, Any]:
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations (id, client_id, title, created_at, updated_at) "
                "VALUES (?,?,?,?,?)",
                (conv_id, client_id, "", now, now),
            )
            self._conn.commit()
        return {"id": conv_id, "title": "", "created_at": now, "updated_at": now}

    def _owns_conversation(self, conv_id: str, client_id: str) -> bool:
        """会话归属校验：知道 conv_id 不等于有权读写。"""
        row = self._conn.execute(
            "SELECT 1 FROM conversations WHERE id=? AND client_id=?",
            (conv_id, client_id),
        ).fetchone()
        return bool(row)

    def set_conversation_title(
        self, conv_id: str, title: str, *, client_id: str | None = None
    ) -> bool:
        now = _now()
        with self._lock:
            if client_id is not None and not self._owns_conversation(conv_id, client_id):
                return False
            self._conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (title[:40], now, conv_id),
            )
            self._conn.commit()
        return True

    def touch_conversation(self, conv_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conv_id)
            )
            self._conn.commit()

    def append_message(
        self,
        conv_id: str,
        role: str,
        content: dict[str, Any],
        *,
        client_id: str | None = None,
    ) -> bool:
        with self._lock:
            if client_id is not None and not self._owns_conversation(conv_id, client_id):
                return False
            self._conn.execute(
                "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?,?,?,?)",
                (conv_id, role, json.dumps(content, ensure_ascii=False), _now()),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conv_id)
            )
            self._conn.commit()
        return True

    def load_messages(
        self, conv_id: str, *, client_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            if client_id is not None and not self._owns_conversation(conv_id, client_id):
                return []
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
                (conv_id,),
            ).fetchall()
        return [
            {"role": r["role"], "content": json.loads(r["content"] or "{}")}
            for r in rows
        ]

    def delete_conversation(
        self, conv_id: str, *, client_id: str | None = None
    ) -> bool:
        with self._lock:
            if client_id is not None and not self._owns_conversation(conv_id, client_id):
                return False
            self._conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
            self._conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
            self._conn.commit()
        return True

    def delete_empty_conversations(self, client_id: str, *, keep_id: str = "") -> int:
        """删除该用户下没有消息的会话（keep_id 保留），返回删除条数。

        历史上每次刷新都可能新建空会话，侧栏虽已过滤隐藏，
        DB 里却一直堆积。这里提供手动清理入口。
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE client_id=? AND id<>? "
                "AND NOT EXISTS "
                "(SELECT 1 FROM messages m WHERE m.conversation_id=conversations.id)",
                (client_id, keep_id),
            )
            self._conn.commit()
            return cur.rowcount

    def save_plan_with_event(
        self,
        client_id: str,
        data: dict[str, Any],
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        """规划落库 + 画像事件一次事务写入。

        两个来源（plans 表与 profile_events）必须同时成功：
        只写一半会出现「看板有规划、画像没更新」的错位。
        """
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO plans (client_id, plan_json, updated_at) VALUES (?,?,?) "
                    "ON CONFLICT(client_id) DO UPDATE SET "
                    "plan_json=excluded.plan_json, updated_at=excluded.updated_at",
                    (client_id, json.dumps(data, ensure_ascii=False), _now()),
                )
                self._conn.execute(
                    "INSERT INTO profile_events (client_id, kind, payload, ts) "
                    "VALUES (?,?,?,?)",
                    (
                        client_id,
                        kind,
                        json.dumps(payload, ensure_ascii=False),
                        _now(),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # -- 当前规划 -------------------------------------------------------

    def save_plan(self, client_id: str, data: dict[str, Any]) -> None:
        """保存当前规划（每用户一份，覆盖写）。

        规划必须落库：只存 session_state 的话，刷新就丢，
        就无法判断「用户是否已经制定过规划」这个流程前提。
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO plans (client_id, plan_json, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(client_id) DO UPDATE SET "
                "plan_json=excluded.plan_json, updated_at=excluded.updated_at",
                (client_id, json.dumps(data, ensure_ascii=False), _now()),
            )
            self._conn.commit()

    def load_plan(self, client_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT plan_json FROM plans WHERE client_id=?", (client_id,)
            ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["plan_json"] or "{}")
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) and data else None

    def delete_plan(self, client_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM plans WHERE client_id=?", (client_id,))
            self._conn.commit()

    # -- 周期学习总结 ---------------------------------------------------

    def has_summary(self, client_id: str, period_type: str, period_date: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM summaries WHERE client_id=? AND period_type=? AND period_date=?",
                (client_id, period_type, period_date),
            ).fetchone()
        return row is not None

    def save_summary(
        self, client_id: str, period_type: str, period_date: str, content: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO summaries "
                "(client_id, period_type, period_date, content, generated_at) "
                "VALUES (?,?,?,?,?)",
                (client_id, period_type, period_date, content, _now()),
            )
            self._conn.commit()

    def list_summaries(
        self, client_id: str, period_type: str
    ) -> list[dict[str, str]]:
        """该用户某类总结，按周期日期倒序。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT period_date, content, generated_at FROM summaries "
                "WHERE client_id=? AND period_type=? ORDER BY period_date DESC",
                (client_id, period_type),
            ).fetchall()
        return [
            {
                "period_date": r["period_date"],
                "content": r["content"],
                "generated_at": r["generated_at"],
            }
            for r in rows
        ]


_store: ProfileStore | None = None
_store_lock = threading.Lock()


def get_profile_store() -> ProfileStore:
    """进程级单例（参照 notebook/store.py 的 get_store）。"""
    global _store
    with _store_lock:
        if _store is None:
            _store = ProfileStore()
    return _store
