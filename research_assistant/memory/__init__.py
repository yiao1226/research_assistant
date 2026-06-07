"""科研助手 — 多层记忆系统。

组件:
  - WorkingMemory: 会话级工作记忆（内存，防抖队列触发压缩）
  - EpisodicMemory: 跨会话情景记忆（SQLite + Qdrant）
  - SemanticMemory: 语义记忆（Neo4j 知识图谱）
  - UserProfileManager: 用户画像管理（结构化事实 + 被动注入）
  - + Fact Extraction: 结构化事实提取（置信度、分类、增量合并）
  - + Message Processing: 纠正/认可信号检测
  - + Debounce Queue: 防抖记忆更新队列

参考: DeerFlow 记忆系统架构
"""
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .knowledge_graph import KnowledgeGraph
from .user_profile import UserProfileManager
from .fact_extraction import (
    extract_facts_from_conversation,
    merge_facts,
    format_facts_for_injection,
    build_user_profile_text,
)
from .message_processing import (
    detect_correction,
    detect_reinforcement,
    filter_user_messages,
    classify_turn,
)
from .debounce_queue import (
    MemoryUpdateQueue,
    MemoryQueueEntry,
    get_memory_queue,
    reset_memory_queue,
)
