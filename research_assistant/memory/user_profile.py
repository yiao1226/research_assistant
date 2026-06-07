"""用户画像管理 — 被动注入机制。

参考 DeerFlow 的 DynamicContextMiddleware，在每次 QA 对话开始前:
  1. 从已有结构化事实中构建用户画像
  2. 从情景记忆中提取未解决问题
  3. 格式化为 <user_profile> 注入到 LLM 上下文

与 DeerFlow 的区别:
  - DeerFlow: 注入到 LangGraph 消息列表（ID 替换技巧）
  - Project2: 注入到 intent prompt 的 context 字段（更简单，效果等价）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 用户画像 JSON 文件路径（per-user）
FACTS_FILENAME = "user_facts.json"
PROFILE_FILENAME = "profile.json"


class UserProfileManager:
    """管理用户的结构化画像数据。

    存储:
      - 结构化事实: data/users/{name}/user_facts.json
        {version, lastUpdated, userContext: {...}, facts: [{content, category, confidence, ...}]}
      - 基础画像: data/users/{name}/profile.json (已有，user_manager 管理)
    """

    def __init__(self, username: str, user_dir: str = ""):
        self.username = username
        if user_dir:
            self._user_dir = Path(user_dir)
        else:
            self._user_dir = Path("./data/users") / username

    # ── 路径 ──

    @property
    def facts_path(self) -> Path:
        return self._user_dir / FACTS_FILENAME

    # ── 读写 ──

    def load_facts(self) -> dict:
        """加载用户的结构化事实。

        Returns:
            {"version": "1.0", "lastUpdated": "...",
             "userContext": {...}, "facts": [{...}]}
        """
        if not self.facts_path.exists():
            return {
                "version": "1.0",
                "lastUpdated": "",
                "userContext": {},
                "facts": [],
            }

        try:
            with open(self.facts_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("加载用户事实失败: %s", e)
            return {"version": "1.0", "lastUpdated": "", "userContext": {}, "facts": []}

    def save_facts(self, data: dict) -> bool:
        """保存用户结构化事实（原子写入）。"""
        from datetime import datetime

        data["version"] = "1.0"
        data["lastUpdated"] = datetime.now().isoformat()

        self._user_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 原子写入: 临时文件 → rename
            import uuid
            tmp_path = self.facts_path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self.facts_path)
            logger.info("用户事实已保存: %s (%d 条)", self.username, len(data.get("facts", [])))
            return True
        except OSError as e:
            logger.error("保存用户事实失败: %s", e)
            return False

    def get_facts_list(self) -> list[dict]:
        """获取事实列表（便捷方法）。"""
        return self.load_facts().get("facts", [])

    def get_user_context(self) -> dict:
        """获取用户上下文摘要。"""
        return self.load_facts().get("userContext", {})

    def update_facts(self, user_context: dict, new_facts: list[dict],
                     facts_to_remove: list[str] | None = None) -> bool:
        """更新用户事实（增量合并 + 原子写入）。

        Args:
            user_context: 新的用户上下文
            new_facts: 新提取的事实
            facts_to_remove: 要删除的事实 ID 列表

        Returns:
            是否保存成功
        """
        from .fact_extraction import merge_facts, remove_facts

        current = self.load_facts()

        # 合并上下文
        merged_context = dict(current.get("userContext", {}))
        for key, value in user_context.items():
            if value:  # 非空才更新
                merged_context[key] = value

        # 合并事实
        current_facts = current.get("facts", [])
        if facts_to_remove:
            current_facts = remove_facts(current_facts, facts_to_remove)
        merged_facts = merge_facts(current_facts, new_facts)

        return self.save_facts({
            "userContext": merged_context,
            "facts": merged_facts,
        })

    def build_context_for_qa(self, unresolved_questions: list[str] | None = None) -> str:
        """构建用于 QA 注入的用户画像上下文。

        Args:
            unresolved_questions: 从情景记忆获取的未解决问题

        Returns:
            格式化的 <user_profile> 文本，可直接拼入 intent prompt
        """
        from .fact_extraction import build_user_profile_text

        data = self.load_facts()
        return build_user_profile_text(
            facts=data.get("facts", []),
            user_context=data.get("userContext", {}),
            unresolved_questions=unresolved_questions,
        )

    # ── 统计 ──

    def stats(self) -> dict:
        data = self.load_facts()
        facts = data.get("facts", [])
        categories = {}
        for f in facts:
            cat = f.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_facts": len(facts),
            "categories": categories,
            "has_context": bool(data.get("userContext", {}).get("researchFocus")),
            "last_updated": data.get("lastUpdated", ""),
        }
