"""核心模块单测。

覆盖四类**容易出错且出错代价高**的逻辑：
    1. 防泄漏隔离 —— 错了会让准确率虚高且极难发现
    2. chunk_id 唯一性 —— 撞了会在 Qdrant 静默丢数据
    3. 指标计算 —— 手算样例对照
    4. 引用校验与押题扫描 —— 红线，必须为 0
"""

from __future__ import annotations

import math

import pytest

from cs408rag.eval.metrics import (
    aggregate,
    evaluate_query,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from cs408rag.generation.citation import (
    check_citations,
    detect_predictions,
    extract_citations,
    extract_exam_refs,
)
from cs408rag.ingestion.chunker import (
    Chunk,
    assert_no_holdout_leak,
    assert_unique_ids,
    chunk_explanations,
    split_text,
)
from cs408rag.ingestion.quiz_loader import Question, split_by_year, tag_frequency
from cs408rag.retrieval.fusion import (
    contribution_stats,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_recall_basic(self):
        assert recall_at_k(["a", "b", "c"], {"a"}, 1) == 1.0
        assert recall_at_k(["b", "a"], {"a"}, 1) == 0.0
        assert recall_at_k(["b", "a"], {"a"}, 2) == 1.0
        assert recall_at_k(["a", "b", "c", "d"], {"a", "c"}, 2) == 0.5
        assert recall_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 1.0

    def test_recall_empty_relevant(self):
        """没有 ground truth 时返回 0，而不是除零崩掉。"""
        assert recall_at_k(["a"], set(), 1) == 0.0

    def test_ndcg_hand_computed(self):
        # 相关项 {a}，[a,b,c] -> DCG=1/log2(2)=1, IDCG=1
        assert abs(ndcg_at_k(["a", "b", "c"], {"a"}, 3) - 1.0) < 1e-9
        # 相关项 {a,b}，[b,x,a] -> DCG=1/log2(2)+1/log2(4)=1.5
        #                      IDCG=1/log2(2)+1/log2(3)
        expect = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert abs(ndcg_at_k(["b", "x", "a"], {"a", "b"}, 3) - expect) < 1e-9

    def test_mrr(self):
        assert abs(reciprocal_rank(["x", "y", "a"], {"a"}) - 1 / 3) < 1e-9
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_aggregate_excludes_no_gt(self):
        """★ 无 ground truth 的题不能计入分母 —— 那会让指标既不准也不可比。"""
        ms = [
            evaluate_query("q1", ["a", "b"], {"a"}),
            evaluate_query("q2", ["c"], {"c"}),
            evaluate_query("q3", ["z"], set()),  # 无 GT
        ]
        agg = aggregate(ms, ks=(5, 10))
        assert agg.n_queries == 3
        assert agg.n_queries_with_relevant == 2
        assert agg.recall_at[10] == 1.0  # 只对有效的 2 题求平均

    def test_aggregate_latency(self):
        ms = [evaluate_query("q1", ["a"], {"a"})]
        agg = aggregate(ms, latencies_ms=[10.0, 20.0, 30.0], ks=(5,))
        assert agg.avg_latency_ms == 20.0
        assert agg.p50_latency_ms == 20.0


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------


class TestFusion:
    def test_rrf_consensus_wins(self):
        """多路共识的文档应当排在单路独有的文档之前。"""
        r = reciprocal_rank_fusion(
            {"a": ["x", "consensus"], "b": ["consensus", "y"]}, k=60
        )
        assert r[0].doc_id == "consensus"
        assert r[0].ranks == {"a": 2, "b": 1}

    def test_rrf_k_smooths(self):
        """k 越大，高排名与低排名的差距越小。"""
        big_k = reciprocal_rank_fusion({"a": ["x", "y"]}, k=1000)
        small_k = reciprocal_rank_fusion({"a": ["x", "y"]}, k=1)
        assert (small_k[0].score / small_k[1].score) > (big_k[0].score / big_k[1].score)

    def test_rrf_weights(self):
        r = reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, weights={"a": 2.0, "b": 1.0})
        assert r[0].doc_id == "x"

    def test_rrf_rejects_bad_k(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({"a": ["x"]}, k=0)

    def test_weighted_normalizes_across_scales(self):
        """加权融合要先归一化，否则量纲大的那路会碾压其他路。"""
        r = weighted_score_fusion(
            {"dense": [("a", 0.9), ("b", 0.1)], "sparse": [("c", 1000.0), ("d", 1.0)]},
            normalize="minmax",
        )
        scores = {x.doc_id: x.score for x in r}
        # minmax 后每路最高分都是 1.0，所以 a 与 c 并列
        assert abs(scores["a"] - scores["c"]) < 1e-9

    def test_contribution_stats(self):
        rankings = {"dense": ["a", "b"], "graph": ["c"]}
        fused = reciprocal_rank_fusion(rankings)
        st = contribution_stats(rankings, fused, top_n=3)
        assert st["dense"]["in_top_n"] == 2
        assert st["graph"]["in_top_n"] == 1


# ---------------------------------------------------------------------------
# 防泄漏与数据完整性（红线）
# ---------------------------------------------------------------------------


def _q(year: int, num: int, tags: list[str] | None = None) -> Question:
    return Question(
        year=year,
        num=num,
        subject="数据结构",
        stem=f"题干{year}-{num}",
        options={"A": "1", "B": "2"},
        answer="A",
        tags=tags or ["测试"],
        explanation=f"解析{year}-{num}",
    )


class TestLeakageGuard:
    def test_holdout_explanations_blocked(self):
        qs = [_q(2020, 1), _q(2021, 2), _q(2024, 3), _q(2025, 4)]
        chunks, blocked = chunk_explanations(qs, holdout_years={2023, 2024, 2025, 2026})
        assert len(chunks) == 2  # 2020, 2021 入库
        assert len(blocked) == 2  # 2024, 2025 被挡
        assert {c.year for c in chunks} == {2020, 2021}

    def test_assert_detects_leak(self):
        """★ 万一有 held-out 解析混进待入库列表，必须炸掉而不是放行。"""
        bad = Chunk(
            chunk_id="x",
            text="t",
            subject="数据结构",
            source="explanation",
            year=2024,
        )
        with pytest.raises(RuntimeError, match="防泄漏校验失败"):
            assert_no_holdout_leak([bad], holdout_years={2023, 2024})

    def test_assert_passes_for_dev(self):
        ok = Chunk(chunk_id="x", text="t", subject="数据结构", source="explanation", year=2020)
        assert_no_holdout_leak([ok], holdout_years={2023, 2024})

    def test_assert_unique_ids_catches_collision(self):
        """★ chunk_id 撞了会在 Qdrant 静默覆盖丢数据，必须在这里拦住。"""
        cs = [
            Chunk(chunk_id="dup", text="a", subject="X", source="wangdao"),
            Chunk(chunk_id="dup", text="b", subject="X", source="wangdao"),
        ]
        with pytest.raises(RuntimeError, match="chunk_id 冲突"):
            assert_unique_ids(cs)

    def test_assert_unique_ids_ok(self):
        cs = [
            Chunk(chunk_id="a", text="a", subject="X", source="wangdao"),
            Chunk(chunk_id="b", text="b", subject="X", source="wangdao"),
        ]
        assert_unique_ids(cs)


class TestQuizLoader:
    def test_split_no_overlap_no_loss(self):
        qs = [_q(y, 1) for y in range(2009, 2027)]
        sp = split_by_year(qs, holdout_years={2023, 2024, 2025, 2026})
        assert len(sp.dev) + len(sp.holdout) == len(qs)
        assert not ({q.year for q in sp.dev} & {q.year for q in sp.holdout})

    def test_tag_frequency(self):
        qs = [_q(2020, 1, ["A", "B"]), _q(2020, 2, ["A"])]
        f = tag_frequency(qs)
        assert f["A"] == 2 and f["B"] == 1


class TestChunker:
    def test_split_short_text_unchanged(self):
        assert split_text("短文本", chunk_size=100) == ["短文本"]

    def test_split_long_text_respects_size(self):
        text = "句子。" * 400
        parts = split_text(text, chunk_size=200, overlap=0)
        assert len(parts) > 1
        # 允许一定超出（在句子边界断开），但不能离谱
        assert all(len(p) <= 400 for p in parts)

    def test_split_empty(self):
        assert split_text("") == []
        assert split_text("   ") == []


# ---------------------------------------------------------------------------
# 引用与押题（红线）
# ---------------------------------------------------------------------------


class TestCitation:
    def test_extract_forms(self):
        assert extract_citations("见 [1] 和 [2]。") == [1, 2]
        assert extract_citations("[1][1][3]") == [1, 3]
        assert extract_citations("【1】和［2］") == [1, 2]
        assert extract_citations("无引用") == []

    def test_extract_exam_refs(self):
        assert extract_exam_refs("如 2011#21、2015-33") == ["2011#21", "2015#33"]

    def test_precision_penalizes_fabrication(self):
        class B:
            def __init__(self, i):
                self.index = i

        rep = check_citations("见 [1][9]。", [B(1), B(2)])
        assert rep.valid_indices == [1]
        assert rep.invalid_indices == [9]
        assert abs(rep.precision - 0.5) < 1e-9

    @pytest.mark.parametrize(
        "text",
        ["今年可能考这个", "预测明年会出这类题", "大概率会考", "这是必考点"],
    )
    def test_prediction_detected(self, text):
        """★ 押题是红线，必须能被扫出来。"""
        assert detect_predictions(text)

    def test_normal_analysis_not_flagged(self):
        assert detect_predictions("该考点历年考查 23 次，如 2011#21、2015#33。") == []

    def test_fabricated_exam_ref_flagged(self):
        class B:
            def __init__(self, i):
                self.index = i

        rep = check_citations(
            "如 2099#99 所示。", [B(1)], valid_exam_refs={"2011#21"}
        )
        assert rep.invalid_exam_refs == ["2099#99"]


# ---------------------------------------------------------------------------
# 教材插图链路
# ---------------------------------------------------------------------------


class TestImages:
    def test_extract_img_tags_with_caption(self):
        from cs408rag.ingestion.md_loader import _extract_images

        raw = (
            "### 5.1.2 基本术语\n"
            "　　下面结合图 5.1 中的树来说明。\n"
            '<div align="center"><img src="./pictures/abc123.jpg" alt="插图"></div>\n'
            '<p align="center"><em>图 5.1 树的树形表示</em></p>\n'
        )
        imgs = _extract_images(raw, section_id="s1", subject="数据结构", base_dir=".")
        assert len(imgs) == 1
        assert imgs[0]["caption"] == "图 5.1 树的树形表示"
        assert imgs[0]["src"] == "./pictures/abc123.jpg"
        # image_id 带科目前缀，避免不同科目的短文件名相撞
        assert imgs[0]["image_id"].startswith("数据结构-")
        # anchor 用于把图挂到「它所在的那一段」
        assert "基本术语" in imgs[0]["anchor"] or "图 5.1" in imgs[0]["anchor"]

    def test_extract_skips_external_urls(self):
        from cs408rag.ingestion.md_loader import _extract_images

        raw = '<img src="https://example.com/a.jpg"><img src="./pictures/b.jpg">'
        imgs = _extract_images(raw, section_id="s", subject="操作系统", base_dir=".")
        assert len(imgs) == 1
        assert imgs[0]["src"] == "./pictures/b.jpg"

    def test_images_assigned_to_matching_chunk_only(self):
        """★ 图片要挂到它出现的那一段所在 chunk，而不是整节所有 chunk。

        否则一个切成 5 段的小节会让每个 chunk 都带同一批图，
        检索结果里塞满无关插图。
        """
        from cs408rag.ingestion.chunker import chunk_md_sections
        from cs408rag.ingestion.md_loader import MdSection

        body = "前段讲概念。" * 60 + "这里插入图。" + "后段讲别的。" * 60
        sec = MdSection(
            section_id="s1",
            subject="数据结构",
            chapter="第一章",
            section="一节",
            title="一节",
            path=["第一章", "一节"],
            level=2,
            text=body,
            source_file="x.md",
            images=[
                {
                    "image_id": "数据结构-abc",
                    "src": "./pictures/abc.jpg",
                    "caption": "",
                    "anchor": "这里插入图。",
                }
            ],
        )
        chunks = chunk_md_sections([sec])
        assert len(chunks) > 1, "测试前提：该节应被切成多段"
        with_img = [c for c in chunks if c.images]
        assert len(with_img) == 1, f"图片应只挂在一个 chunk 上，实际 {len(with_img)} 个"
        assert with_img[0].images[0]["image_id"] == "数据结构-abc"

    def test_empty_section_with_images_does_not_crash(self):
        """只有图片、没有正文的小节不应抛异常（实测踩过 IndexError）。"""
        from cs408rag.ingestion.chunker import chunk_md_sections
        from cs408rag.ingestion.md_loader import MdSection

        sec = MdSection(
            section_id="s2",
            subject="数据结构",
            chapter="",
            section="空",
            title="空",
            path=["空"],
            level=2,
            text="",
            source_file="x.md",
            images=[{"image_id": "数据结构-x", "src": "./pictures/x.jpg", "caption": "", "anchor": ""}],
        )
        assert chunk_md_sections([sec]) == []

    def test_resolve_images_missing_index_is_safe(self):
        """索引缺失时应返回 url 为空而不是抛异常 —— 缺图不该阻断检索。"""
        from cs408rag.ingestion.image_index import resolve_images

        out = resolve_images([{"image_id": "根本不存在的id", "caption": "某图"}])
        assert len(out) == 1
        assert out[0]["url"] == ""
        assert out[0]["exists"] is False

    def test_resolve_none(self):
        from cs408rag.ingestion.image_index import resolve_images

        assert resolve_images(None) == []
        assert resolve_images([]) == []