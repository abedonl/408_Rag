"""用户画像：提问/规划事件存储 + 纯规则统计。

明确取舍（与 notebook/store.py 一致）：
- 不引入账号系统，用匿名 client_id（前端写进 URL 查询串 ?cid=）。
- 只存**事件**（question / plan），画像是读取时按需聚合的派生视图，
  单一事实来源，不存在「事件与画像不一致」的问题。
- 标签提取是纯规则（子串 + 最长片段匹配），不调 LLM，结果可人工核对。
"""

from cs408rag.profile.stats import build_profile, extract_tags
from cs408rag.profile.store import ProfileStore, get_profile_store

__all__ = ["ProfileStore", "build_profile", "extract_tags", "get_profile_store"]
