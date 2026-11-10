"""会话记忆：内存 LRU + TTL，不跨会话、不落盘。

明确取舍（Agent v2 plan Week 2）：
- 重启丢历史，这是有意的；不做 SQLite / LangGraph checkpointer / 长期向量记忆。
- 超过 MAX_TURNS 轮时把更早的轮次压成 rule-based summary（不调 LLM），
  prompt 只带 summary + 最近 RECENT_TURNS 轮，避免 token 随会话膨胀。
- SessionStore 独立于 AgentTools —— 工具实例持有 trace 且每次 run 会 reset，
  会话历史挂在工具上会被清掉（plan 中明确警告过这一点）。
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

MAX_SESSIONS = 200
SESSION_TTL_S = 2 * 3600  # 2 小时不活跃即过期
MAX_TURNS = 8  # 会话内最多保留的完整轮次，更早的压进 summary
RECENT_TURNS = 4  # 注入 prompt 的完整轮次数
ANSWER_CLIP = 300  # 历史里每轮回答的截断长度
SUMMARY_MAX_CHARS = 800


@dataclass
class Turn:
    """一轮对话。存原始问句（不是改写后的），答案截断保存。"""

    question: str
    answer: str
    route: str = ""
    qid: str | None = None
    topic: str | None = None
    tool_names: list[str] = field(default_factory=list)
    ts: float = 0.0

    def render(self) -> str:
        a = self.answer[:ANSWER_CLIP]
        if len(self.answer) > ANSWER_CLIP:
            a += "…"
        return f"用户：{self.question}\n助手：{a}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "qid": self.qid,
            "topic": self.topic,
            "tool_names": self.tool_names,
            "ts": self.ts,
        }


@dataclass
class AgentSession:
    session_id: str
    client_id: str = ""  # 归属身份；空串表示未绑定（API 未传 client_id 的场景）
    turns: list[Turn] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    last_qid: str | None = None
    last_topic: str | None = None
    summary: str = ""
    turn_count: int = 0  # 含被压进 summary 的轮次，用于 turn_no
    created_at: float = 0.0
    updated_at: float = 0.0

    def render_history(self) -> str:
        """summary + 最近 RECENT_TURNS 轮，供 prompt 注入；空会话返回空串。"""
        parts: list[str] = []
        if self.summary:
            parts.append(f"（更早的对话摘要）\n{self.summary}")
        for t in self.turns[-RECENT_TURNS:]:
            parts.append(t.render())
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "turn_count": self.turn_count,
            "profile": self.profile,
            "last_qid": self.last_qid,
            "last_topic": self.last_topic,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class InMemorySessionStore:
    """LRU + TTL 的内存会话表。所有方法线程安全。"""

    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl_s: float = SESSION_TTL_S):
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self._sessions: OrderedDict[str, AgentSession] = OrderedDict()
        self._lock = Lock()

    def get_or_create(
        self,
        session_id: str | None = None,
        profile: dict[str, Any] | None = None,
        client_id: str = "",
    ) -> AgentSession:
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            sid = session_id or uuid.uuid4().hex[:12]
            s = self._sessions.get(sid)
            if s is not None and s.client_id and client_id and s.client_id != client_id:
                # 同一 session_id 被另一个身份使用：既不读也不写别人的历史，
                # 改为返回一个独立会话（新 id，不覆盖原会话）。
                sid = uuid.uuid4().hex[:12]
                s = None
            if s is None:
                s = AgentSession(
                    session_id=sid,
                    client_id=client_id,
                    created_at=now,
                    updated_at=now,
                )
                self._sessions[sid] = s
            else:
                # 首次绑定身份（如 API 建会话、UI 后续带 cid 续聊）
                if client_id and not s.client_id:
                    s.client_id = client_id
                s.updated_at = now
                self._sessions.move_to_end(sid)
            if profile:
                s.profile.update(profile)
            return s

    def get(self, session_id: str) -> AgentSession | None:
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            s = self._sessions.get(session_id)
            if s is not None:
                self._sessions.move_to_end(session_id)
            return s

    def append_turn(self, session: AgentSession, turn: Turn) -> int:
        """追加一轮并做裁剪；返回该轮的 turn_no（从 1 开始，含被压缩的轮次）。"""
        with self._lock:
            turn.ts = turn.ts or time.time()
            session.turns.append(turn)
            session.turn_count += 1
            session.updated_at = time.time()
            if turn.qid:
                session.last_qid = turn.qid
            if turn.topic:
                session.last_topic = turn.topic
            while len(session.turns) > MAX_TURNS:
                self._compress_oldest(session)
            return session.turn_count

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            self._prune_locked(time.time())
            return len(self._sessions)

    # -- 内部 ----------------------------------------------------------------

    @staticmethod
    def _compress_oldest(session: AgentSession) -> None:
        """把最早的一轮压成一行摘要。规则压缩，不调 LLM。"""
        old = session.turns.pop(0)
        a = old.answer.replace("\n", " ")[:60]
        line = f"· 问过「{old.question[:30]}」（{old.route or 'qa'}）→ {a}"
        session.summary = (session.summary + "\n" + line if session.summary else line)
        if len(session.summary) > SUMMARY_MAX_CHARS:
            # 超限时从头部按行丢弃，保留最近的摘要
            lines = session.summary.splitlines()
            while lines and sum(len(x) for x in lines) > SUMMARY_MAX_CHARS:
                lines.pop(0)
            session.summary = "\n".join(lines)

    def _prune_locked(self, now: float) -> None:
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.updated_at > self.ttl_s
        ]
        for sid in expired:
            del self._sessions[sid]
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)  # 逐出最久未用


_store: InMemorySessionStore | None = None


def get_session_store() -> InMemorySessionStore:
    global _store
    if _store is None:
        _store = InMemorySessionStore()
    return _store
