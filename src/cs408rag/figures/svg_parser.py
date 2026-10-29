"""SVG 图形解析（图表题的 T1 / T2 层）。

109 道真题含图，其中 72 道题干里有内联 SVG。408 的图有一个很有利的性质：
它们大多是 draw.io 导出的**结构图**（二叉树、图、流水线、存储层次），
`<ellipse cx cy>` / `<rect x y>` 是节点，`<path d="M x y l dx dy">` 是边，
坐标足以还原拓扑。

这个模块不依赖多模态模型，纯文本解析，因此：

- **能拿到**：节点位置、边的连接关系、连通分量（一张图里有 4 棵独立的树
  = 这题在问「下列哪个是…」）、树高/平衡性/是否完全二叉树
- **拿不到**：draw.io 把节点文字渲染成了 base64 位图（实测 59/72 题），
  所以节点标签通常不可读。这个限制必须在输出里明说，不能让 LLM 以为
  它看到了「值为 3,1,7 的节点」。

README / 评测报告里必须把这件事讲清楚：T1 的贡献是**拓扑**，不是 OCR。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------

RE_SVG_OPEN = re.compile(r"<svg\b[^>]*>", re.I)
RE_TAG = re.compile(r"<([A-Za-z][\w:.-]*)((?:\s+[^<>]*?)?)/?>", re.S)
RE_ATTR = re.compile(r'([\w:.-]+)\s*=\s*"([^"]*)"')
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
RE_TEXT_BLOCK = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S | re.I)
RE_TSPAN = re.compile(r"<tspan\b[^>]*>(.*?)</tspan>", re.S | re.I)
RE_TAG_STRIP = re.compile(r"<[^>]+>")


def extract_outer_svgs(html: str) -> list[str]:
    """从 HTML 片段里提取**最外层** SVG，嵌套的 SVG 不重复计入。

    draw.io 导出里 `<svg>` 可以嵌套（外层画布 + 内层图形），
    用非贪婪正则 `.*?</svg>` 会把外层截断，导致坐标不完整。这里做深度匹配。
    """
    out: list[str] = []
    i = 0
    lower = html.lower()
    while True:
        start = lower.find("<svg", i)
        if start < 0:
            break
        # 找匹配的 </svg>
        depth = 0
        j = start
        end = -1
        while j < len(html):
            nxt_open = lower.find("<svg", j)
            nxt_close = lower.find("</svg>", j)
            if nxt_close < 0:
                break
            if 0 <= nxt_open < nxt_close:
                depth += 1
                j = nxt_open + 4
            else:
                depth -= 1
                j = nxt_close + 6
                if depth == 0:
                    end = j
                    break
        if end < 0:
            out.append(html[start:])
            break
        out.append(html[start:end])
        i = end
    return out


def _attrs(tag: str) -> dict[str, str]:
    return {k.lower(): v for k, v in RE_ATTR.findall(tag)}


def _f(attrs: dict[str, str], key: str, default: float = 0.0) -> float:
    v = attrs.get(key)
    if v is None:
        return default
    m = RE_NUM.search(v)
    return float(m.group(0)) if m else default


def _clean_text(raw: str) -> str:
    s = RE_TAG_STRIP.sub("", raw or "")
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class FigureNode:
    index: int
    x: float
    y: float
    rx: float = 0.0
    ry: float = 0.0
    w: float = 0.0
    h: float = 0.0
    shape: str = "ellipse"
    label: str = ""
    image_label: bool = False  # 节点里有内嵌位图，文字不可读
    component: int = -1

    @property
    def radius(self) -> float:
        if self.shape == "ellipse":
            return max(self.rx, self.ry)
        return max(self.w, self.h) / 2


@dataclass
class FigureEdge:
    x1: float
    y1: float
    x2: float
    y2: float
    a: int | None = None
    b: int | None = None

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class ParsedFigure:
    width: float = 0.0
    height: float = 0.0
    nodes: list[FigureNode] = field(default_factory=list)
    edges: list[FigureEdge] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    n_images: int = 0
    n_paths: int = 0

    # -- 拓扑 -------------------------------------------------------------

    @property
    def components(self) -> list[list[int]]:
        """连通分量（节点索引分组）。并查集实现。"""
        n = len(self.nodes)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in self.edges:
            if e.a is not None and e.b is not None and e.a != e.b:
                ra, rb = find(e.a), find(e.b)
                if ra != rb:
                    parent[rb] = ra

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        # ★ 按最左坐标排序，而不是按节点数。
        # 选项图（A/B/C/D 四棵树）必须按图中从左到右输出，
        # 否则模型会把结构3 当成 C —— 实测 2009-04 因此选错。
        return sorted(groups.values(), key=lambda g: min(self.nodes[i].x for i in g))

    def adjacency(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {i: [] for i in range(len(self.nodes))}
        for e in self.edges:
            if e.a is not None and e.b is not None and e.a != e.b:
                adj[e.a].append(e.b)
                adj[e.b].append(e.a)
        return adj

    def _root_of_component(self, comp: list[int]) -> int:
        """树的根：无父节点、且位置最靠上的节点。"""
        comp_set = set(comp)
        adj = self.adjacency()
        degrees = {i: len([x for x in adj[i] if x in comp_set]) for i in comp}
        # 度为 1 的节点里，y 最小的更容易是根（二叉树画法根在上方）
        cands = [i for i in comp if degrees.get(i, 0) <= 2]
        if not cands:
            cands = comp
        return min(cands, key=lambda i: (self.nodes[i].y, self.nodes[i].x))

    def _tree_profile(self, comp: list[int]) -> dict[str, Any]:
        """若该分量是树，算高度、是否平衡二叉树、是否完全二叉树。

        ★ 必须按**有根树**遍历。无向图里二叉树的内部节点度数为 3
        （父 + 两个子），直接把邻居当子节点会得出「有 3 个孩子 -> 非二叉 ->
        所有树都不平衡」的结论。实测 2009-04（四棵 AVL 候选树）因此全判为
        「不平衡」，而正确答案 B 恰好是其中一棵平衡的。
        """
        comp_set = set(comp)
        adj = {i: [x for x in self.adjacency()[i] if x in comp_set] for i in comp}
        root = self._root_of_component(comp)

        parent: dict[int, int | None] = {root: None}
        children: dict[int, list[int]] = {i: [] for i in comp}
        levels: list[list[int]] = []
        queue: list[tuple[int, int]] = [(root, 0)]
        while queue:
            u, d = queue.pop(0)
            while len(levels) <= d:
                levels.append([])
            levels[d].append(u)
            for v in adj[u]:
                if v == parent[u] or v in parent:
                    continue
                parent[v] = u
                children[u].append(v)
                queue.append((v, d + 1))

        is_tree = len(parent) == len(comp) and len(self.edges_in(comp)) == len(comp) - 1
        # 同一层按横坐标排序，输出结果才和图上从左到右一致
        for lv in levels:
            lv.sort(key=lambda i: self.nodes[i].x)

        binary = is_tree and max((len(c) for c in children.values()), default=0) <= 2
        h_memo: dict[int, int] = {}

        def h(u: int) -> int:
            if u not in h_memo:
                h_memo[u] = 1 + (max((h(c) for c in children[u]), default=0))
            return h_memo[u]

        balanced = None
        complete = None
        if binary:
            balanced = all(
                abs(h(children[u][0]) - h(children[u][1])) <= 1
                if len(children[u]) == 2
                else h(children[u][0]) <= 1
                if len(children[u]) == 1
                else True
                for u in comp
            )
            complete = self._is_complete(root, children)

        return {
            "nodes": len(comp),
            "edges": len(self.edges_in(comp)),
            "is_tree": is_tree,
            "binary": binary,
            "height": h(root) if is_tree else None,
            "root": root,
            "balanced_binary": balanced,
            "complete_binary": complete,
            "max_degree": max((len(adj[i]) for i in comp), default=0),
            "levels": [[_node_label(self.nodes[i]) for i in lv] for lv in levels],
        }

    def edges_in(self, comp: list[int]) -> list[FigureEdge]:
        s = set(comp)
        return [e for e in self.edges if e.a in s and e.b in s]

    @staticmethod
    def _is_complete(root: int, children: dict[int, list[int]]) -> bool:
        """完全二叉树：按层序编号，遇到空位后不能再出现节点。"""
        queue: list[int | None] = [root]
        seen_null = False
        while queue:
            u = queue.pop(0)
            if u is None:
                seen_null = True
                continue
            if seen_null:
                return False
            kids = children.get(u, [])
            queue.extend(kids)
            if len(kids) < 2:
                queue.append(None)
        return True

    # -- 描述 -------------------------------------------------------------

    def describe(self) -> str:
        """生成给 LLM 的文本描述。

        描述里必须显式写出「节点标签是否可读」——否则 LLM 会假设自己
        看到了节点值，从而给出没有依据的结论。
        """
        if not self.nodes and not self.texts:
            return "该图未能解析出任何结构（可能是纯位图或非常规 SVG）。"

        lines = [f"图形解析（SVG 坐标重建）：画布 {self.width:.0f}×{self.height:.0f}，"
                 f"节点 {len(self.nodes)} 个，边 {len(self.edges)} 条。"]
        bound = [e for e in self.edges if e.a is not None and e.b is not None]
        if self.edges and len(bound) < len(self.edges) * 0.5:
            lines.append(
                f"⚠️ 只有 {len(bound)}/{len(self.edges)} 条连线能匹配到节点，"
                "该图可能是流程图或位图叠加，拓扑结构不可靠，不要据此做结构推断。"
            )
        comps = [c for c in self.components if len(c) >= 2]
        big = [c for c in comps if len(c) >= 3]

        if len(big) >= 2:
            lines.append(
                f"检测到 {len(big)} 个相互独立的结构，按图中**从左到右**排列，"
                "通常依次对应选项 A、B、C、D（若题干另有说明以题干为准）："
            )
            for k, comp in enumerate(big, 1):
                prof = self._tree_profile(comp)
                xs = [self.nodes[i].x for i in comp]
                lines.append(
                    f"  结构{k}（x≈{min(xs):.0f}-{max(xs):.0f}）：" + _profile_line(prof)
                )
        elif big:
            prof = self._tree_profile(big[0])
            lines.append("结构：" + _profile_line(prof))
            if prof["is_tree"]:
                lines.append("层级：" + " | ".join(
                    "层{}: {}".format(i, ", ".join(lv)) for i, lv in enumerate(prof["levels"])
                ))
        elif comps:
            lines.append(f"检测到 {len(comps)} 个两两相连的小结构。")

        if self.nodes and len(self.nodes) <= 12:
            coords = " ".join(
                f"N{i+1}({n.x:.0f},{n.y:.0f})" for i, n in enumerate(self.nodes)
            )
            lines.append(f"节点坐标：{coords}")
        elif self.nodes:
            # 坐标对判断选项顺序有用，但 20 个节点的坐标会占掉 prompt 预算，
            # 而结构摘要已经按从左到右给出，足够 LLM 对应 A/B/C/D。
            lines.append(
                f"节点数 {len(self.nodes)} 较多，省略逐点坐标"
                "（结构摘要已按从左到右排列）。"
            )

        if self.texts:
            uniq = list(dict.fromkeys(t for t in self.texts if t))[:30]
            lines.append("图中文本：" + "；".join(uniq))
        else:
            lines.append("图中没有可提取的文本标签。")

        image_nodes = sum(1 for n in self.nodes if n.image_label)
        if self.n_images:
            lines.append(
                f"⚠️ 该 SVG 内含 {self.n_images} 张内嵌位图"
                + (f"（{image_nodes} 个节点标签是位图）" if image_nodes else "")
                + "，draw.io 把文字渲染成了图片，**节点值不可读**。"
                "如需读取节点上的具体数值/字母，需要多模态视觉模型（T3）。"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_images": self.n_images,
            "components": self.components,
            "nodes": [
                {
                    "i": n.index,
                    "x": n.x,
                    "y": n.y,
                    "shape": n.shape,
                    "label": n.label,
                    "image_label": n.image_label,
                    "component": n.component,
                }
                for n in self.nodes
            ],
            "edges": [
                {"a": e.a, "b": e.b, "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2}
                for e in self.edges
            ],
            "texts": self.texts,
            "structure_profiles": [
                self._tree_profile(c) for c in self.components if len(c) >= 3
            ],
        }


def _node_label(n: FigureNode) -> str:
    if n.label:
        return n.label
    return f"N{n.index + 1}"


def _profile_line(p: dict[str, Any]) -> str:
    s = f"{p['nodes']} 个节点、{p['edges']} 条边"
    if p["is_tree"]:
        s += f"，树结构，高度 {p['height']}"
        if p["binary"]:
            s += "，二叉"
            if p["balanced_binary"] is not None:
                s += "，平衡=" + ("是" if p["balanced_binary"] else "否")
            if p["complete_binary"] is not None:
                s += "，完全=" + ("是" if p["complete_binary"] else "否")
    else:
        s += "，非树结构（可能含环或为一般图）"
    return s


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

# 节点外的装饰性 rect（draw.io 的透明边界框）要过滤掉
def _is_decor_rect(attrs: dict[str, str]) -> bool:
    return (
        attrs.get("fill", "").lower() == "none"
        and attrs.get("stroke", "").lower() == "none"
    )


def _path_endpoints(d: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """从 path 的 d 属性里取首尾坐标。支持 M/L/H/V/C/Q/S/T 及其相对形式。"""
    # draw.io 会把 200 压成 2e2（科学计数法），正则必须认指数，
    # 否则 `M2e2 80` 会被切成 M2 / e2 —— e2 被当成未知命令，整条边的坐标全错。
    tokens = re.findall(
        r"[MmLlHhVvCcQqSsTtZz]|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", d or ""
    )
    if not tokens:
        return None
    x = y = 0.0
    start: tuple[float, float] | None = None
    last: tuple[float, float] | None = None
    cmd = ""
    i = 0

    def take(n: int) -> list[float] | None:
        nonlocal i
        if i + n > len(tokens):
            return None
        out = []
        for _ in range(n):
            try:
                out.append(float(tokens[i]))
            except ValueError:
                return None
            i += 1
        return out

    while i < len(tokens):
        t = tokens[i]
        if re.match(r"^[A-Za-z]$", t):
            cmd = t
            i += 1
            if cmd in "Zz":
                if start is not None:
                    last = start
                continue
        if cmd in "Mm":
            v = take(2)
            if not v:
                break
            if cmd == "m":
                x, y = x + v[0], y + v[1]
            else:
                x, y = v[0], v[1]
            start = (x, y)
            last = (x, y)
            cmd = "l" if cmd == "m" else "L"
        elif cmd in "Ll":
            v = take(2)
            if not v:
                break
            x, y = (x + v[0], y + v[1]) if cmd == "l" else (v[0], v[1])
            last = (x, y)
        elif cmd in "Hh":
            v = take(1)
            if not v:
                break
            x = x + v[0] if cmd == "h" else v[0]
            last = (x, y)
        elif cmd in "Vv":
            v = take(1)
            if not v:
                break
            y = y + v[0] if cmd == "v" else v[0]
            last = (x, y)
        elif cmd in "Cc":
            v = take(6)
            if not v:
                break
            x, y = (x + v[4], y + v[5]) if cmd == "c" else (v[4], v[5])
            last = (x, y)
        elif cmd in "Qq":
            v = take(4)
            if not v:
                break
            x, y = (x + v[2], y + v[3]) if cmd == "q" else (v[2], v[3])
            last = (x, y)
        elif cmd in "SsTt":
            v = take(4 if cmd in "Ss" else 2)
            if not v:
                break
            if cmd in "s":
                x, y = x + v[-2], y + v[-1]
            else:
                x, y = v[-2], v[-1]
            last = (x, y)
        else:
            i += 1

    if start is None or last is None:
        return None
    return start, last


def parse_svg(svg: str) -> ParsedFigure:
    """解析一个 SVG 片段。容错优先：任何单个元素失败都不影响整体。"""
    m = RE_SVG_OPEN.search(svg)
    root_attrs = _attrs(m.group(0)) if m else {}
    width = _f(root_attrs, "width")
    height = _f(root_attrs, "height")
    vb = root_attrs.get("viewbox", "")
    if (not width or not height) and vb:
        parts = [p for p in re.split(r"[\s,]+", vb) if p]
        if len(parts) == 4:
            try:
                width = width or float(parts[2])
                height = height or float(parts[3])
            except ValueError:
                pass

    fig = ParsedFigure(width=width, height=height)
    nodes: list[FigureNode] = []
    edges: list[FigureEdge] = []
    texts: list[str] = []
    n_images = 0
    n_paths = 0

    # 逐标签扫描。先收集节点，再收集边（边端点匹配需要节点坐标）。
    for tm in RE_TAG.finditer(svg):
        tag = tm.group(1).lower()
        attrs = _attrs(tm.group(0))

        if tag in ("ellipse", "circle"):
            rx = _f(attrs, "rx") or _f(attrs, "r")
            ry = _f(attrs, "ry") or _f(attrs, "r")
            if rx <= 0 or ry <= 0:
                continue
            nodes.append(
                FigureNode(
                    index=-1,
                    x=_f(attrs, "cx"),
                    y=_f(attrs, "cy"),
                    rx=rx,
                    ry=ry,
                    shape="ellipse",
                )
            )
        elif tag == "rect":
            if _is_decor_rect(attrs):
                continue
            w = _f(attrs, "width")
            h = _f(attrs, "height")
            if w < 8 or h < 8:
                continue
            # draw.io 的 `<rect>` 也可能是整张图的背景板
            if width and height and w >= width * 0.95 and h >= height * 0.95:
                continue
            nodes.append(
                FigureNode(
                    index=-1,
                    x=_f(attrs, "x") + w / 2,
                    y=_f(attrs, "y") + h / 2,
                    w=w,
                    h=h,
                    shape="rect",
                )
            )
        elif tag == "image":
            n_images += 1
        elif tag == "path":
            n_paths += 1
            # 只取「边」：有描边、无填充。draw.io 用 pointer-events=stroke 标记
            if attrs.get("pointer-events", "").lower() not in ("", "stroke"):
                continue
            if attrs.get("fill", "").lower() not in ("", "none"):
                continue
            pts = _path_endpoints(attrs.get("d", ""))
            if pts:
                (x1, y1), (x2, y2) = pts
                if math.hypot(x2 - x1, y2 - y1) >= 5:
                    edges.append(FigureEdge(x1=x1, y1=y1, x2=x2, y2=y2))

    # 文本：text 块里的所有 tspan 合并
    for tb in RE_TEXT_BLOCK.finditer(svg):
        spans = [_clean_text(x) for x in RE_TSPAN.findall(tb.group(1))]
        txt = " ".join(s for s in spans if s) or _clean_text(tb.group(1))
        if txt:
            texts.append(txt)

    # 索引 + 连通分量
    for i, n in enumerate(nodes):
        n.index = i

    fig.nodes = nodes
    fig.edges = edges
    fig.texts = texts
    fig.n_images = n_images
    fig.n_paths = n_paths

    _bind_edges(fig)
    _bind_image_labels(fig, n_images)
    for k, comp in enumerate(fig.components):
        for i in comp:
            fig.nodes[i].component = k
    return fig


def _bind_edges(fig: ParsedFigure) -> None:
    """把边端点匹配到最近的节点。阈值取节点半径的 2 倍与 35px 的较大者。"""
    for e in fig.edges:
        best_a, best_b = None, None
        da, db = 1e18, 1e18
        for n in fig.nodes:
            lim = max(n.radius * 2.0, 35.0)
            d1 = math.hypot(n.x - e.x1, n.y - e.y1)
            d2 = math.hypot(n.x - e.x2, n.y - e.y2)
            if d1 <= lim and d1 < da:
                da, best_a = d1, n.index
            if d2 <= lim and d2 < db:
                db, best_b = d2, n.index
        if best_a is not None and best_b is not None and best_a != best_b:
            e.a, e.b = best_a, best_b


def _bind_image_labels(fig: ParsedFigure, n_images: int) -> None:
    """标记「节点标签是位图」的节点。

    draw.io 会把文本渲染成小尺寸 `<image>`（实测 20–40px 宽）叠在节点上。
    超过画布 30% 的图视为整图位图，不绑定到单个节点。
    """
    if not n_images or not fig.nodes:
        return
    for n in fig.nodes:
        # 无法从 image 标签取精确坐标（已在解析时只计数），保守地认为：
        # 位图数量与节点数量相当时，节点标签大概率是位图。
        if n_images >= max(1, len(fig.nodes) // 2):
            n.image_label = True


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def parse_question_figure(stem_html: str) -> ParsedFigure | None:
    """从题干 HTML 里解析第一幅最外层 SVG。没有 SVG 返回 None。"""
    svgs = extract_outer_svgs(stem_html or "")
    if not svgs:
        return None
    # 取面积最大的那个（嵌套时外层通常最大）
    parsed = [parse_svg(s) for s in svgs]
    return max(parsed, key=lambda f: f.width * f.height)


def describe_question_figure(stem_html: str) -> dict[str, Any]:
    """给工具层用的完整结果。"""
    svgs = extract_outer_svgs(stem_html or "")
    if not svgs:
        return {"tier": "none", "ok": False, "description": "题干中没有内联 SVG。"}

    fig = max((parse_svg(s) for s in svgs), key=lambda f: f.width * f.height)
    bound = sum(1 for e in fig.edges if e.a is not None and e.b is not None)
    # T1 要求拓扑真的重建出来了：节点 + 至少一半的边绑上了节点。
    # 只看「有节点也有边」会把流程图误判成 T1（实测 2012-40：
    # 2 个方框 + 7 条箭头，箭头端点全是图中间的自由坐标，一条都绑不上）。
    has_structure = len(fig.nodes) >= 2 and bound >= 1 and bound >= len(fig.edges) * 0.5
    if has_structure:
        tier = "T1"
    elif fig.texts:
        tier = "T2"
    elif fig.n_images:
        tier = "T3"  # 图和文字都是位图，需要视觉模型
    else:
        tier = "none"
    return {
        "tier": tier,
        "ok": has_structure,
        "description": fig.describe(),
        "figure": fig.to_dict(),
        "n_svgs": len(svgs),
        "bound_edges": bound,
    }