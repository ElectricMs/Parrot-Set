"""
会话记忆（Session Memory）

本模块提供一个极简的“后端会话存储”实现，用于 /agent/message 多轮对话：

- 以 session_id 为 key，保存最近 N 轮历史（history）、最近一次识别物种（last_species）、
  以及最近一次 analyze 的结构化产物（last_analyze_artifacts）。
- 支持 TTL：长时间不活跃的会话会被自动清理。

重要限制：
- 这是“进程内内存 store”：服务重启会丢失；多进程/多 worker 不共享。
- 如果要生产可用，请替换为 Redis / 数据库，并在并发/序列化上做更严格的处理。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SessionState:
    """
    单个会话的状态对象（MVP 版本）。

    字段说明：
    - history: (role, text) 列表，用于保留最近对话上下文
    - last_species: 最近一次识别出的 top1 物种名（用于“它/刚才那只...”追问绑定）
    - last_analyze_artifacts: 最近一次识图的结构化产物（用于前端展示/调试/后续推理）
    """
    session_id: str
    updated_at: float = field(default_factory=lambda: time.time())
    history: List[Tuple[str, str]] = field(default_factory=list)  # (role, text)
    last_species: Optional[str] = None
    last_analyze_artifacts: Optional[Dict[str, Any]] = None


class SessionMemoryStore:
    """
    In-memory session store with TTL and history trimming.
    Notes:
    - This is process-local (will reset on restart, and won't share across workers).
    - Good enough for MVP; can be swapped to Redis later.
    """

    def __init__(self, ttl_seconds: int = 3600, max_turns: int = 10):
        # ttl_seconds: 会话不活跃超过该秒数即视为过期
        # max_turns: 最多保留的“轮数”（当前按 message 计数，大致等于 2*max_turns 条记录）
        self.ttl_seconds = int(ttl_seconds)
        self.max_turns = int(max_turns)
        self._store: Dict[str, SessionState] = {}

    def _now(self) -> float:
        return time.time()

    def _expired(self, state: SessionState) -> bool:
        # 判断是否过期：以 updated_at（最后触碰时间）为准
        return (self._now() - state.updated_at) > self.ttl_seconds

    def _gc(self) -> None:
        # 垃圾回收：清理过期 session（惰性触发：在 get_or_create/size 时清理）
        expired = [sid for sid, st in self._store.items() if self._expired(st)]
        for sid in expired:
            self._store.pop(sid, None)

    def size(self) -> int:
        # 便于 debug/观测当前进程持有多少会话
        self._gc()
        return len(self._store)

    def new_session_id(self) -> str:
        # 用 uuid4 hex 生成 session_id（不含 '-'，适合前端存储与传输）
        return uuid.uuid4().hex

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        """
        获取 session；若不存在或已过期则创建新的。

        说明：
        - 若传入 session_id 但已过期：会删除旧记录并创建新记录（沿用该 session_id 或生成新 id）
        """
        self._gc()
        if session_id and session_id in self._store:
            st = self._store[session_id]
            if not self._expired(st):
                return st
            self._store.pop(session_id, None)

        sid = session_id or self.new_session_id()
        st = SessionState(session_id=sid)
        self._store[sid] = st
        return st

    def touch(self, st: SessionState) -> None:
        # 触碰会话：刷新更新时间，防止 TTL 回收
        st.updated_at = self._now()

    def append(self, st: SessionState, role: str, text: str) -> None:
        """
        写入一条历史消息，并按 max_turns 做裁剪。

        当前裁剪策略：
        - 以“消息条数”裁剪，而不是严格的“轮数”对象；
          上限为 max_turns*2（约等于用户+助手各一条为一轮）。
        """
        if not text:
            return
        st.history.append((role, text))
        # Keep only last N turns (each turn is user+agent, but we trim by messages for simplicity)
        if len(st.history) > self.max_turns * 2:
            st.history = st.history[-self.max_turns * 2 :]
        self.touch(st)

    def set_last_species(self, st: SessionState, species: Optional[str]) -> None:
        # 更新最近识别物种名（用于追问绑定）
        st.last_species = species
        self.touch(st)

    def set_last_analyze_artifacts(self, st: SessionState, artifacts: Optional[Dict[str, Any]]) -> None:
        # 保存最近一次识图 artifacts（可能较大，生产环境可只保存摘要）
        st.last_analyze_artifacts = artifacts
        self.touch(st)


