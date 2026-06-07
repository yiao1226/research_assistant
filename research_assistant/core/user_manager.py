"""用户管理 — 身份识别、切换、隔离数据空间。

每个用户拥有独立的:
  - data/users/{username}/library.db  (SQLite)
  - data/users/{username}/checkpoints.db (LangGraph)
  - data/users/{username}/backups/     (JSON 快照)
  - Qdrant collection 前缀 user_{username}_papers / _progress / _memory
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .storage import PerUserStorage


class User:
    """用户模型。"""
    def __init__(self, username: str, base_dir: str | Path):
        self.username = username
        self.user_dir = Path(base_dir) / "data" / "users" / username
        self.storage: Optional[PerUserStorage] = None  # lazy init

    def init_storage(self):
        if self.storage is None:
            self.storage = PerUserStorage(self.user_dir)

    @property
    def profile_path(self):
        return self.user_dir / "profile.json"

    def get_profile(self) -> dict:
        if self.profile_path.exists():
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"username": self.username, "created_at": datetime.now().isoformat()}

    def save_profile(self, data: dict):
        self.user_dir.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def paper_count(self) -> int:
        self.init_storage()
        return self.storage.count_papers()

    def progress_count(self) -> int:
        self.init_storage()
        return len(self.storage.get_all_progress(limit=1000))

    def __repr__(self):
        return f"User({self.username})"


class UserManager:
    """全局用户管理器。"""

    def __init__(self, base_dir: str | Path = "."):
        self.base_dir = Path(base_dir)
        self.users_dir = self.base_dir / "data" / "users"
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self._current_user: Optional[User] = None

    @property
    def current_user(self) -> Optional[User]:
        return self._current_user

    @property
    def is_logged_in(self) -> bool:
        return self._current_user is not None

    def user_exists(self, username: str) -> bool:
        """检查用户是否存在。"""
        username = self._sanitize(username)
        return (self.users_dir / username / "profile.json").exists()

    def login(self, username: str, create_if_missing: bool = True) -> User:
        """登录（不存在时可选择是否创建）。"""
        username = self._sanitize(username)
        user = User(username, self.base_dir)
        exists = self.user_exists(username)

        if not exists:
            if not create_if_missing:
                raise ValueError(f"用户 '{username}' 不存在。")
            user.user_dir.mkdir(parents=True, exist_ok=True)
            user.save_profile({
                "username": username,
                "created_at": datetime.now().isoformat(),
                "last_login": datetime.now().isoformat(),
            })
        else:
            p = user.get_profile()
            p["last_login"] = datetime.now().isoformat()
            user.save_profile(p)
        user.init_storage()
        self._current_user = user
        return user

    def switch_user(self, username: str) -> User:
        return self.login(username)

    def delete_user(self, username: str) -> bool:
        """删除用户及其全部数据（SQLite、Qdrant、文件）。

        Returns:
            True 删除成功，False 用户不存在。
        """
        username = self._sanitize(username)
        if not self.user_exists(username):
            return False

        import shutil

        # 1) 删除本地文件
        user_dir = self.users_dir / username
        if user_dir.exists():
            shutil.rmtree(user_dir)

        # 2) 删除 Qdrant collections（直接用 client，避免触发 EmbeddingService）
        try:
            from qdrant_client import QdrantClient
            import os
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=url)
            for ctype in ["papers", "progress", "memory"]:
                col_name = f"user_{username}_{ctype}"
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
        except Exception:
            pass

        # 3) 如果删除的是当前用户，清空登录状态
        if self._current_user and self._current_user.username == username:
            self._current_user = None

        return True

    def list_users(self) -> list[str]:
        """列出所有注册用户。"""
        if not self.users_dir.exists():
            return []
        users = []
        for d in self.users_dir.iterdir():
            if d.is_dir() and (d / "profile.json").exists():
                users.append(d.name)
        return sorted(users)

    def get_last_session_context(self) -> dict | None:
        """获取用户上次会话的上下文摘要。"""
        if not self._current_user:
            return None
        user = self._current_user
        user.init_storage()
        # 最近搜索历史
        recent_searches = user.storage.get_search_history(limit=5)
        # 最近进展
        recent_progress = user.storage.get_all_progress(limit=5)
        # 论文数
        paper_count = user.storage.count_papers()

        return {
            "recent_searches": [s.get("query", "") for s in recent_searches],
            "recent_progress": [
                {"title": p.get("title", ""), "topic": p.get("topic", ""), "date": p.get("timestamp", "")}
                for p in recent_progress
            ],
            "paper_count": paper_count,
            "last_login": user.get_profile().get("last_login", ""),
        }

    def _sanitize(self, name: str) -> str:
        """净化用户名，只保留字母数字下划线。"""
        import re
        name = name.strip().lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        return name[:30] or "default"
