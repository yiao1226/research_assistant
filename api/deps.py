"""依赖注入 — 复用现有 CLI 业务逻辑，不重写。"""

from __future__ import annotations
import sys, os
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from research_assistant.core import UserManager
from research_assistant.core.storage import PerUserStorage

# 全局单例
_user_manager: UserManager | None = None


def get_user_manager() -> UserManager:
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager(".")
    return _user_manager


def get_storage(username: str):
    mgr = get_user_manager()
    user = mgr.login(username)
    user.init_storage()
    return user.storage


def get_qa_service(username: str):
    """获取指定用户的 QA 服务实例。复用 cli/commands.py 的 get_qa 逻辑。"""
    mgr = get_user_manager()
    mgr.login(username)
    # 注入到 commands 模块
    import cli.commands as cmd_mod
    cmd_mod.user_manager = mgr
    from research_assistant.agent import QAService
    storage = get_storage(username)
    return QAService(username, storage)
