"""CLI 显示 — Banner 和帮助文本。"""

BANNER = r"""
==================================================
  科研助手 - Research Assistant
  LangChain + LangGraph + Qdrant + Neo4j
  智能问答 | 语义检索 | 三层记忆 | 个性化分析
=================================================="""

HELP_TEXT = """
=== 科研助手 ===
ask <问题>                         智能问答（基于知识库）
search [-s arxiv|s2|wos] 关键词   个性化论文搜索
upload <pdf> | inbox scan/status  上传论文
download <arxiv_id>               下载论文PDF
review <主题>                      文献综述
record <主题>                      记录进展
progress <主题>                    查看进展
recall <问题>                      回忆历史
backup                            备份数据
user list | user switch <name>    用户管理
paper list | delete <id> | rebuild <id>  知识库管理
"""

COMMAND_LIST = """
可用命令:
  search [-s arxiv|s2|wos] 关键词  — 个性化论文搜索
  upload <pdf> | upload inbox scan — 上传论文
  download <arxiv_id>              — 下载论文PDF
  review <主题>                     — 文献综述工作流
  record <主题>                     — 记录实验进展
  progress <主题>                   — 查看研究进展
  ask <问题>                        — 智能问答（基于知识库）
  recall <问题>                     — 回忆历史记忆
  user list | switch | delete <name> — 用户管理
  backup                           — 备份数据
  help                             — 帮助
  quit                             — 退出
"""


def print_banner():
    print(BANNER)
