"""作答判分：对照 questions.json 的 ground truth，维护错题本。

只判题库内的题——720 道真题带 answer 标注，判分有据可依。临时粘贴、
无法定位 qid 的题没有 ground truth，返回 known=False 由调用方提示，
不落库（不存没有依据的记录）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cs408rag.ingestion.quiz_loader import Question, load_questions
from cs408rag.notebook.store import NotebookStore, get_store


@lru_cache(maxsize=1)
def _truth_index() -> dict[str, Question]:
    """qid → Question。720 题全量载入是毫秒级，缓存一次即可。"""
    return {q.qid: q for q in load_questions()}


def judge(
    client_id: str,
    qid: str,
    your_choice: str,
    store: NotebookStore | None = None,
) -> dict[str, Any]:
    """判分并维护错题本：答错记录/累加，答对且在则移出（答对自动移出）。"""
    store = store or get_store()
    q = _truth_index().get(qid.strip())
    if q is None or not q.answer:
        return {"known": False, "qid": qid}

    truth = q.answer.strip().upper()
    # subject/tags 两个分支都返回：调用方（API/Streamlit）据此写画像 quiz 事件，
    # 答对时错题本里可能根本没有记录可查。
    base = {"known": True, "qid": q.qid, "truth": truth, "subject": q.subject, "tags": q.tags}
    if your_choice.strip().upper() == truth:
        removed = store.remove(client_id, q.qid)
        return {**base, "correct": True, "removed": removed}

    item = store.record_wrong(
        client_id,
        q.qid,
        your_choice=your_choice.strip().upper(),
        correct_choice=truth,
        subject=q.subject,
        tags=q.tags,
    )
    return {**base, "correct": False, "recorded": True, "item": item}
