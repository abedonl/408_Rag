"""图表题分层降级读取（T1 → T2 → T3）。

408 的图表题按定义不是检索问题：检索只能返回文字，看不懂图。
这里的策略是**分层降级**，每一层独立可用，不依赖下一层：

    T1  SVG 坐标重建 —— 节点/边/连通分量/树高/平衡性，纯文本解析，零成本
    T2  SVG 文本提取 —— 结构弱但图里有文字（表格、流程说明）
    T3  多模态视觉   —— 整图是位图时的唯一办法，**可插拔 Provider**

T3 的真实状态：真题图绝大多数是 draw.io 导出，节点文字被渲染成 base64 位图
（实测 59/72 题）。本机没有 SVG→PNG 渲染器，也没配置 VLM，所以 T3 当前
**不可用**。这不是留白，是如实标注：`available=False` + 明确原因，
上层（Agent / 评测）据此把这类题单独归因，不能算进检索或 T1 的账上。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from loguru import logger

from cs408rag.config import RAW_DIR, settings
from cs408rag.figures.svg_parser import describe_question_figure

# ---------------------------------------------------------------------------
# T3：可插拔视觉 Provider
# ---------------------------------------------------------------------------


class VisionProvider(Protocol):
    """多模态视觉接口。实现方需把图片和问题转成文本描述。"""

    name: str

    def available(self) -> bool: ...

    def describe(self, image_bytes: bytes, question: str = "") -> str: ...


class OpenAIVisionProvider:
    """OpenAI 兼容的多模态接口（百炼兼容模式同款协议）。"""

    name = "openai-vision"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.vlm_model
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url

    def available(self) -> bool:
        # 还需要能把 SVG 渲染成 PNG —— 本机没装 cairosvg 时同样不可用
        return bool(self.model and self.api_key) and _renderer_available()

    def describe(self, image_bytes: bytes, question: str = "") -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=90)
        b64 = base64.b64encode(image_bytes).decode()
        prompt = (
            "这是一道 408 考研题的配图。请只描述图中**客观可见**的结构："
            "图形类型、节点/元素数量、连接关系、标注文字。"
            "不要解题，不要推断图中没有的内容。"
        )
        if question:
            prompt += f"\n题目：{question[:300]}"
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip()


def _renderer_available() -> bool:
    try:
        import cairosvg  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


_provider: VisionProvider | None = None


def get_vision_provider() -> VisionProvider | None:
    global _provider
    if _provider is None:
        p = OpenAIVisionProvider()
        _provider = p if p.available() else None
    return _provider


# ---------------------------------------------------------------------------
# 题干 SVG 缓存
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _stem_html_by_qid() -> dict[str, str]:
    """qid -> 题干 HTML。Question 模型不保留 HTML，图表解析需要原始 SVG。"""
    path = RAW_DIR / "quiz" / "questions.json"
    if not path.exists():
        logger.warning(f"找不到真题库 {path}，read_figure 将不可用")
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for q in raw:
        qid = f"{int(q['year'])}-{int(q['num']):02d}"
        out[qid] = q.get("stem_html") or ""
        # 解析里的图经常包含答案示例，只在题干无图时兜底；此处不取，
        # 避免把含答案的图喂给模型（防泄漏）。
    logger.info(f"图表题题干索引：{len(out)} 题")
    return out


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class FigureResult:
    qid: str
    tier: str  # T1 / T2 / T3 / none
    available: bool
    description: str
    detail: dict[str, Any]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "tier": self.tier,
            "available": self.available,
            "description": self.description,
            "detail": self.detail,
            "note": self.note,
        }


def read_figure(qid: str, *, question: str = "") -> FigureResult:
    """分层读取图表题的图。

    Args:
        qid: 真题编号，如 `2009-04`
        question: 题干文本，仅在 T3 视觉描述时作为提示
    """
    from cs408rag.agent.exam_analysis import normalize_qid

    qid = normalize_qid(qid)
    html = _stem_html_by_qid().get(qid, "")
    if not html:
        return FigureResult(
            qid=qid,
            tier="none",
            available=False,
            description="",
            detail={},
            note=f"找不到 {qid} 的题干数据。",
        )

    parsed = describe_question_figure(html)
    tier = parsed["tier"]

    if tier in ("T1", "T2"):
        return FigureResult(
            qid=qid,
            tier=tier,
            available=True,
            description=parsed["description"],
            detail={"figure": parsed.get("figure"), "bound_edges": parsed.get("bound_edges")},
            note=(
                "T1 重建了拓扑，但节点上的文字多为位图，结论只应基于结构"
                "（如树高、平衡性），不要假设读到了节点值。"
                if tier == "T1"
                else "T2 只能读到图中文本，没有拓扑结构。"
            ),
        )

    # T3：位图。尝试视觉 Provider。
    if tier == "none":
        return FigureResult(
            qid=qid,
            tier="none",
            available=False,
            description="",
            detail={},
            note=(
                "题干中没有内联图形（该题的图只出现在解析里，或 svg_count 统计的是"
                "解析部分）。read_figure 不读取解析中的图，避免把含答案的示例图喂给模型。"
            ),
        )

    provider = get_vision_provider()
    if provider is None:
        return FigureResult(
            qid=qid,
            tier="T3",
            available=False,
            description=parsed["description"],
            detail={"n_images": (parsed.get("figure") or {}).get("n_images", 0)},
            note=(
                "该题的图是内嵌位图，需要多模态视觉模型才能读取。"
                "当前未配置可用的 VLM（VLM_MODEL）或 SVG 渲染器（cairosvg），"
                "本层不可用。评测中此类题单独归因，不计入检索或 T1 的收益。"
            ),
        )

    image = _render_svg_to_png(html)
    if not image:
        return FigureResult(
            qid=qid,
            tier="T3",
            available=False,
            description=parsed["description"],
            detail={},
            note="视觉 Provider 已配置，但 SVG 渲染失败（缺少 cairosvg 或系统依赖）。",
        )
    try:
        text = provider.describe(image, question)
        return FigureResult(
            qid=qid,
            tier="T3",
            available=True,
            description=text,
            detail={"provider": provider.name},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"视觉描述失败：{e}")
        return FigureResult(
            qid=qid,
            tier="T3",
            available=False,
            description=parsed["description"],
            detail={},
            note=f"视觉模型调用失败：{str(e)[:150]}",
        )


def _render_svg_to_png(html: str) -> bytes | None:
    """把题干里的第一幅 SVG 渲染成 PNG。没有渲染器时返回 None。"""
    if not _renderer_available():
        return None
    try:
        import cairosvg

        from cs408rag.figures.svg_parser import extract_outer_svgs

        svgs = extract_outer_svgs(html)
        if not svgs:
            return None
        return cairosvg.svg2png(bytestring=svgs[0].encode(), output_width=1000)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"SVG 渲染失败：{e}")
        return None


def figure_coverage() -> dict[str, int]:
    """统计全部含图题的分层覆盖，供 README / API 展示。"""
    from collections import Counter

    from cs408rag.agent.exam_analysis import ExamAnalyzer

    analyzer = ExamAnalyzer()
    c: Counter[str] = Counter()
    for q in analyzer.questions:
        if not q.has_figure:
            continue
        r = read_figure(q.qid, question=q.stem)
        c[r.tier] += 1
    return dict(c)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    for qid in ("2009-03", "2009-04", "2011-27", "2012-40"):
        r = read_figure(qid)
        print(f"--- {qid} tier={r.tier} available={r.available}")
        print(r.description[:400])
        if r.note:
            print("NOTE:", r.note)
    print("\n覆盖：", figure_coverage())