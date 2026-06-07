"""文档加载器 — 多格式支持。

策略（经中英论文全篇实测验证）:
  PDF:      PyMuPDF 提取 → 行清洗 → 伪MD注入 → Markdown
  非 PDF:   MarkItDown 转换 → Markdown（Word/Excel/PPT/图片等）
  纯文本:   直接读取

导出:
  - load_document(): 统一加载入口
  - DocumentLoader: 可配置的加载器类
"""
from .loader import DocumentLoader, load_document
