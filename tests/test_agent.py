"""二期模块单测。

覆盖三类**出错代价高**的逻辑：

    1. 计算器沙箱 —— 放行注入就是远程代码执行
    2. SVG 拓扑重建 —— 结构判错会直接答错图表题（2009-04 实测）
    3. 考点统计口径 —— 主题词聚合错了，备考结论就是错的

不依赖 LLM / Qdrant / Neo4j，纯本地可跑。
"""

from __future__ import annotations

import pytest

from cs408rag.agent.calculator import convert_units, evaluate
from cs408rag.agent.exam_analysis import ExamAnalyzer, classify_type, normalize_qid, qid_to_ref
from cs408rag.figures.svg_parser import (
    _path_endpoints,
    describe_question_figure,
    extract_outer_svgs,
)
from cs408rag.generation.prompt import parse_choice
from cs408rag.ingestion.quiz_loader import Question

# ---------------------------------------------------------------------------
# 计算器
# ---------------------------------------------------------------------------


class TestCalculator:
    def test_basic_arithmetic(self):
        assert evaluate("2**10")["value"] == 1024
        assert evaluate("(1 + 2) * 3 / 4")["value"] == 2.25
        assert evaluate("log2(4096)")["value"] == 12
        assert evaluate("0b1010 + 0x1F")["value"] == 41

    def test_display_trims_float(self):
        assert evaluate("4/2")["display"] == "2"
        assert evaluate("1/3")["display"] == "0.333333"

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('echo hacked')",
            "open('x')",
            "().__class__.__bases__",
            "[1,2,3]",
            "x = 1",
            "lambda: 1",
            "eval('2')",
        ],
    )
    def test_rejects_non_whitelisted(self, expr):
        assert evaluate(expr)["ok"] is False

    def test_rejects_zero_division(self):
        r = evaluate("1/0")
        assert r["ok"] is False

    def test_rejects_too_long(self):
        assert evaluate("1+" * 400 + "1")["ok"] is False

    def test_unit_conversion(self):
        assert "32768" in convert_units("4KB")["expr"]


# ---------------------------------------------------------------------------
# SVG 解析
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer() -> ExamAnalyzer:
    return ExamAnalyzer()


@pytest.fixture(scope="module")
def figures() -> dict[str, str]:
    import json

    from cs408rag.config import RAW_DIR

    raw = json.loads((RAW_DIR / "quiz" / "questions.json").read_text(encoding="utf-8"))
    return {f"{q['year']}-{q['num']:02d}": q.get("stem_html") or "" for q in raw}


class TestSvgParser:
    def test_scientific_notation_in_path(self):
        """draw.io 用 2e2 表示 200，解析错会让整条边的坐标飞掉。"""
        pts = _path_endpoints("M2e2 80l45.86-45.86")
        assert pts == ((200.0, 80.0), (245.86, 34.14))

    def test_extract_outer_svgs_handles_nesting(self):
        html = "<svg><g><svg><path/></svg></g></svg><svg></svg>"
        assert len(extract_outer_svgs(html)) == 2

    def test_single_binary_tree(self, figures):
        """2009-03：7 节点 / 6 条边，拓扑必须完整。"""
        r = describe_question_figure(figures["2009-03"])
        assert r["tier"] == "T1"
        assert r["figure"]["n_nodes"] == 7
        assert r["figure"]["n_edges"] == 6
        prof = r["figure"]["structure_profiles"][0]
        assert prof["is_tree"] is True
        assert prof["binary"] is True

    def test_option_figures_ordered_left_to_right(self, figures):
        """2009-04：四棵树按从左到右对应 A/B/C/D，平衡的必须是第二棵（B）。"""
        r = describe_question_figure(figures["2009-04"])
        profiles = r["figure"]["structure_profiles"]
        assert len(profiles) == 4
        # 结构按最左 x 排序
        comps = r["figure"]["components"]
        xs = [min(r["figure"]["nodes"][i]["x"] for i in c) for c in comps]
        assert xs == sorted(xs)
        assert profiles[1]["balanced_binary"] is True
        assert sum(1 for p in profiles if p["balanced_binary"]) == 1

    def test_flowchart_degrades_to_t3(self, figures):
        """2012-40 是位图流程图，一条边都绑不上，不能谎称 T1。"""
        r = describe_question_figure(figures["2012-40"])
        assert r["tier"] == "T3"
        assert r["ok"] is False
        assert r["bound_edges"] == 0

    def test_text_layer(self, figures):
        r = describe_question_figure(figures["2011-27"])
        assert r["tier"] == "T2"
        assert "可用资源数" in r["description"]

    def test_no_svg(self):
        assert describe_question_figure("<p>纯文字</p>")["tier"] == "none"


# ---------------------------------------------------------------------------
# 考点统计
# ---------------------------------------------------------------------------


class TestExamAnalysis:
    def test_normalize_qid(self):
        assert normalize_qid("2011#21") == "2011-21"
        assert normalize_qid("2011年第 7 题") == "2011-07"
        assert qid_to_ref("2011-21") == "2011#21"

    def test_exam_whitelist_format_matches_extractor(self):
        """白名单与引用抽取必须用同一种编号格式，否则真实引用会被判成编造。"""
        from cs408rag.generation.citation import extract_exam_refs
        from cs408rag.generation.service import build_exam_ref_whitelist

        wl = build_exam_ref_whitelist()
        assert "2011#21" in wl
        assert all(extract_exam_refs(f"见 {r}")[0] in wl for r in list(wl)[:20])

    def test_theme_word_aggregates_tags(self, analyzer):
        """「死锁」应汇总死锁概念/预防/处理/必要条件四个标签，而不是只取一个。"""
        s = analyzer.stats("死锁")
        assert len(s.matched_tags) >= 3
        assert s.total == len(set(s.qids))
        assert s.total >= max(s.by_tag.values())

    def test_exact_tag_not_over_aggregated(self, analyzer):
        s = analyzer.stats("排序算法")
        assert s.matched_tags == ["排序算法"]
        assert s.total == 23

    def test_unknown_tag(self, analyzer):
        assert analyzer.stats("不存在的考点xyz").total == 0

    def test_type_classification(self, analyzer):
        # 2009-01 题干含「计算机」，不能因此被判成计算题
        assert analyzer.type_of["2009-01"] == "concept"
        assert analyzer.type_of["2014-24"] == "calc"
        assert analyzer.type_of["2009-04"] == "figure"

    def test_classify_ignores_computer_word(self):
        q = Question(
            year=2009, num=1, subject="数据结构",
            stem="为解决计算机与打印机之间速度不匹配的问题，设置一个缓冲区。",
            options={}, answer="B",
        )
        assert classify_type(q) == "concept"

    def test_related_tags_have_evidence(self, analyzer):
        rel = analyzer.related_tags("排序算法", top_n=3)
        for r in rel:
            assert r["evidence"], "共现关系必须带证据题号"
            assert r["cooccurrence"] >= 1

    def test_year_allowed_filters(self):
        a = ExamAnalyzer(year_allowed={2009})
        assert all(q.year == 2009 for q in a.questions)


# ---------------------------------------------------------------------------
# 答案解析
# ---------------------------------------------------------------------------


class TestPredictionScan:
    @pytest.mark.parametrize(
        "text",
        [
            "上述归纳仅供参考，不作预测。",
            "不构成对未来考情的任何预测。",
            "本系统不做考试预测。",
        ],
    )
    def test_negated_is_not_a_hit(self, text):
        from cs408rag.generation.citation import detect_predictions

        assert detect_predictions(text) == []

    @pytest.mark.parametrize(
        "text,hit",
        [
            ("预测 2027 年会出此类题。", "预测"),
            ("该考点今年可能考。", "今年可能"),
            ("这题必考。", "必考"),
        ],
    )
    def test_real_prediction_still_caught(self, text, hit):
        from cs408rag.generation.citation import detect_predictions

        assert hit in detect_predictions(text)


class TestRetrievalMetricsExtra:
    def test_hit_vs_recall_with_multi_chunk_gt(self):
        """GT 小节有两个 chunk，只命中一个：Recall=0.5，Accuracy=1。"""
        from cs408rag.eval.metrics import hit_at_k, recall_at_k

        retrieved = ["x", "a", "y"]
        rel = {"a", "b"}
        assert recall_at_k(retrieved, rel, 3) == 0.5
        assert hit_at_k(retrieved, rel, 3) == 1.0
        assert hit_at_k(retrieved, rel, 1) == 0.0

    def test_precision_at_k(self):
        from cs408rag.eval.metrics import precision_at_k

        assert precision_at_k(["a", "b", "x", "y"], {"a", "b"}, 4) == 0.5
        assert precision_at_k(["a", "b", "x", "y"], {"a", "b"}, 2) == 1.0
        # 分母固定为 k：只返回 1 条命中 1 条，P@10 = 0.1 而不是 1.0
        assert precision_at_k(["a"], {"a"}, 10) == 0.1
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_aggregate_exposes_precision_and_accuracy(self):
        from cs408rag.eval.metrics import aggregate, evaluate_query

        ms = [
            evaluate_query("q1", ["a", "x"], {"a", "b"}),
            evaluate_query("q2", ["x", "y"], {"z"}),
        ]
        agg = aggregate(ms, ks=(10,))
        # q1: top10 里命中 1 个 -> P=0.1；q2: 0 -> 平均 0.05
        assert abs(agg.precision_at[10] - 0.05) < 1e-9
        assert abs(agg.hit_at[10] - 0.5) < 1e-9
        row = agg.to_row(ks=(10,))
        assert "Precision@10" in row and "Accuracy@10" in row


class TestSectionDedup:
    def _data(self):
        from cs408rag.retrieval.fusion import FusedItem
        from cs408rag.retrieval.vector_store import SearchHit

        fused = [
            FusedItem(doc_id=x, score=s, ranks={}, raw_scores={})
            for x, s in [("a1", 3.0), ("a2", 2.9), ("b1", 2.8), ("a3", 2.7), ("c1", 2.6)]
        ]
        hits = {
            x: SearchHit(chunk_id=x, score=0, payload={"section_id": sec})
            for x, sec in [("a1", "s1"), ("a2", "s1"), ("b1", "s2"), ("a3", "s1"), ("c1", "s3")]
        }
        return fused, hits

    def test_keeps_one_per_section_in_score_order(self):
        from cs408rag.retrieval.pipeline import dedup_by_section

        fused, hits = self._data()
        out = [i.doc_id for i in dedup_by_section(fused, hits, per_sec=1)]
        assert out == ["a1", "b1", "c1"]

    def test_keeps_two_per_section(self):
        from cs408rag.retrieval.pipeline import dedup_by_section

        fused, hits = self._data()
        out = [i.doc_id for i in dedup_by_section(fused, hits, per_sec=2)]
        assert out == ["a1", "a2", "b1", "c1"]

    def test_production_configs_use_zscore_and_dedup(self):
        from cs408rag.agent.tools import AGENT_RETRIEVAL_CONFIG
        from cs408rag.generation.service import DEFAULT_QA_CONFIG

        for cfg in (DEFAULT_QA_CONFIG, AGENT_RETRIEVAL_CONFIG):
            assert cfg.fusion == "weighted"
            assert cfg.normalize == "zscore"
            assert cfg.section_dedup is True
            # 图谱扩展与节去重收益重叠，生产必须关闭
            assert cfg.use_graph_expand is False


class TestParentExpand:
    def test_dedup_parent_and_skip_explanation(self, monkeypatch):
        import cs408rag.retrieval.parent as parent
        from cs408rag.retrieval.vector_store import SearchHit

        monkeypatch.setattr(
            parent,
            "_parent_texts",
            lambda: {"ch > s": "父节完整讲解文本" * 20},
        )
        hits = [
            SearchHit(
                "a", 1, {"chapter_path": ["ch", "s", "x"], "text": "子块a", "source": "wangdao"}
            ),
            SearchHit(
                "b", 1, {"chapter_path": ["ch", "s", "y"], "text": "子块b", "source": "wangdao"}
            ),
            SearchHit("c", 1, {"chapter_path": [], "text": "解析", "source": "explanation"}),
        ]
        out = parent.expand_to_parent(hits, max_chars_per_hit=500)
        # 同一父节的两条只保留一条；解析 chunk 原样保留
        assert [h.chunk_id for h in out] == ["a", "c"]
        assert out[0].payload["parent_expanded"] is True
        assert out[1].text == "解析"


class TestLLMFallback:
    def test_empty_main_falls_back_to_backup(self, monkeypatch):
        from types import SimpleNamespace

        from cs408rag.generation.llm import LLMClient

        calls = []

        class FakeCompletions:
            def create(self, **kw):
                calls.append((kw["model"], kw["max_tokens"]))
                text = "" if kw["model"] == "main" else "答案：C"
                usage = SimpleNamespace(prompt_tokens=10, completion_tokens=kw["max_tokens"])
                msg = SimpleNamespace(content=text)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=msg)], usage=usage
                )

        client = LLMClient(model="main", backup_model="backup", api_key="k")
        monkeypatch.setattr(
            client,
            "_ensure",
            lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
        )
        resp = client.chat([{"role": "user", "content": "题"}], max_tokens=100)

        assert resp.ok and resp.model == "backup" and resp.text == "答案：C"
        # 主模型先试一次 + 翻倍一次，然后才降级
        assert [m for m, _ in calls] == ["main", "main", "backup"]

    def test_backup_not_used_when_main_succeeds(self, monkeypatch):
        from types import SimpleNamespace

        from cs408rag.generation.llm import LLMClient

        calls = []

        class FakeCompletions:
            def create(self, **kw):
                calls.append(kw["model"])
                usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
                msg = SimpleNamespace(content="答案：B")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=msg)], usage=usage
                )

        client = LLMClient(model="main", backup_model="backup", api_key="k")
        monkeypatch.setattr(
            client,
            "_ensure",
            lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
        )
        resp = client.chat([{"role": "user", "content": "题"}], max_tokens=100)
        assert resp.model == "main"
        assert calls == ["main"]


class TestEvalIsolation:
    def test_exam_stats_strips_answers_in_eval_mode(self):
        from cs408rag.agent.tools import AgentTools

        t = AgentTools(eval_mode=True)
        r = t.call("get_exam_stats", kp_name="排序算法")
        assert r.ok and r.data["total"] > 0
        assert all("answer" not in ex for ex in (r.data.get("examples") or []))

    def test_exam_stats_keeps_answers_in_production(self):
        from cs408rag.agent.tools import AgentTools

        t = AgentTools(eval_mode=False)
        r = t.call("get_exam_stats", kp_name="排序算法")
        assert r.ok
        assert any("answer" in ex for ex in (r.data.get("examples") or []))


class TestParseChoice:
    @pytest.mark.parametrize(
        "text,expect",
        [
            ("答案：B", "B"),
            ("答案：**B**（从左到右第二个）", "B"),
            ("【答案】D", "D"),
            ("**答案：A**", "A"),
            ("答案 B", "B"),
            ("\nC\n", "C"),
            ("没有给出答案", ""),
        ],
    )
    def test_formats(self, text, expect):
        assert parse_choice(text) == expect