"""CLI 显示 — Banner 和帮助文本。"""

BANNER = r"""
==================================================
  科研助手 - Research Assistant
  LangChain + LangGraph + Qdrant + Neo4j
  智能问答 | 语义检索 | 三层记忆 | 个性化分析
=================================================="""

HELP_TEXT = """
=== 科研助手 ===

直接输入问题即可，无需命令前缀。
Agent 会自动判断: 查知识库 / 回忆历史 / 搜索论文 / 闲聊。

快速命令（/ 前缀）:
  /search 或 /s  关键词              个性化论文搜索
  /upload 或 /u  <pdf> | inbox scan  上传论文
  /review 或 /r  <主题>              文献综述工作流
  /record 或 /n  <主题>              记录研究进展
  /progress 或 /p <主题>             查看研究进展
  /download 或 /d <arxiv_id>         下载论文PDF
  /backup                            备份数据
  /user list | switch | delete       用户管理
  /paper list | delete | rebuild      知识库管理
  /ask 或 /a  <问题>                 强制智能问答

无前缀直接拖入 PDF 路径即可分析并入库。

输入 help 查看此帮助，quit 退出。
"""

COMMAND_LIST = """
直接输入问题即可 ── Agent 自动分析 + 工具调用
  /search 或 /s    关键词 — 外部论文搜索
  /review 或 /r    主题 — 文献综述 (Agent驱动)
  /research 或 /rs 主题 — 深度研究分析
  /progress-report 或 /pr 主题 — 进展评估报告
  /upload 或 /u    <pdf>  — 上传论文
  /record 或 /n    主题 — 记录进展
  /progress 或 /p  主题 — 查看进展
  /backup            — 备份数据
  help               — 帮助
  quit               — 退出
"""


def print_banner():
    print(BANNER)
