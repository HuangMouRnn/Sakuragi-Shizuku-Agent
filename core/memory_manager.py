"""Memory manager for cross-agent context sharing and history compression."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import AgentMessage, AgentRole, PipelineState

logger = logging.getLogger(__name__)


class MemorySlot:
    """A single memory entry with metadata."""

    def __init__(self, key: str, value: Any, source: AgentRole, ttl: int | None = None):
        self.key = key
        self.value = value
        self.source = source
        self.created_at = datetime.now()
        self.access_count = 0
        self.ttl = ttl  # max turns before eviction

    def access(self) -> Any:
        self.access_count += 1
        return self.value

    def is_expired(self, current_turn: int) -> bool:
        if self.ttl is None:
            return False
        return self.access_count >= self.ttl


class MemoryManager:
    """Manages shared memory across agents with compression and eviction."""

    def __init__(self, max_slots: int = 200, persist_dir: str | None = None):
        self._slots: dict[str, MemorySlot] = {}
        self._max_slots = max_slots
        self._turn_counter = 0
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._agent_buffers: dict[AgentRole, list[dict]] = defaultdict(list)

    # ── Core CRUD ──

    def put(self, key: str, value: Any, source: AgentRole, ttl: int | None = None):
        if len(self._slots) >= self._max_slots:
            self._evict()
        self._slots[key] = MemorySlot(key, value, source, ttl)
        logger.debug("Memory PUT: %s (from %s)", key, source.value)

    def get(self, key: str, default: Any = None) -> Any:
        slot = self._slots.get(key)
        if slot is None:
            return default
        return slot.access()

    def get_prefix(self, prefix: str) -> dict[str, Any]:
        return {k: s.access() for k, s in self._slots.items() if k.startswith(prefix)}

    def delete(self, key: str):
        self._slots.pop(key, None)

    # ── Pipeline State Helpers ──

    def save_state(self, state: PipelineState):
        self.put("pipeline_state", state.model_dump(), AgentRole.DEBUG)

    def load_state(self) -> PipelineState | None:
        raw = self.get("pipeline_state")
        if raw is None:
            return None
        return PipelineState(**raw)

    # ── Agent Buffers (append-only logs) ──

    def append_log(self, agent: AgentRole, entry: dict[str, Any]):
        entry["turn"] = self._turn_counter
        entry["timestamp"] = datetime.now().isoformat()
        self._agent_buffers[agent].append(entry)

    def get_logs(self, agent: AgentRole, last_n: int | None = None) -> list[dict]:
        logs = self._agent_buffers[agent]
        if last_n is not None:
            return logs[-last_n:]
        return logs

    # ── Turn Management ──

    def advance_turn(self):
        self._turn_counter += 1
        self._expire_old()

    # ── Eviction ──

    def _evict(self):
        """LRU-style eviction: remove least-accessed, oldest entries."""
        if not self._slots:
            return
        sorted_slots = sorted(
            self._slots.values(),
            key=lambda s: (s.access_count, s.created_at),
        )
        to_remove = max(1, len(sorted_slots) // 10)
        for slot in sorted_slots[:to_remove]:
            del self._slots[slot.key]
        logger.debug("Evicted %d memory slots", to_remove)

    def _expire_old(self):
        expired = [k for k, s in self._slots.items() if s.is_expired(self._turn_counter)]
        for k in expired:
            del self._slots[k]

    # ── Compression ──

    def compress_summarize(self, agent: AgentRole, llm_call=None) -> str | None:
        """Compress agent buffer into a summary. Requires an LLM callable."""
        logs = self._agent_buffers.get(agent, [])
        if len(logs) < 5:
            return None

        recent = logs[-20:]
        text = json.dumps(recent, indent=2, default=str)
        if len(text) < 2000:
            return None

        if llm_call:
            summary = llm_call(
                f"Summarize the following agent log into key decisions and outcomes only:\n\n{text[:6000]}"
            )
        else:
            summary = f"[{len(recent)} entries compressed. Last turn: {recent[-1].get('turn', '?')}]"

        self._agent_buffers[agent] = [{"type": "summary", "content": summary, "turn": self._turn_counter}]
        return summary

    # ── Persistence ──

    def save_to_disk(self, path: str | None = None):
        target = Path(path) if path else self._persist_dir
        if target is None:
            return
        target.mkdir(parents=True, exist_ok=True)
        data = {
            "slots": {k: {"value": s.value, "source": s.source.value} for k, s in self._slots.items()},
            "buffers": {k.value: v for k, v in self._agent_buffers.items()},
            "turn": self._turn_counter,
        }
        (target / "memory.json").write_text(json.dumps(data, indent=2, default=str))

    def load_from_disk(self, path: str | None = None):
        target = Path(path) if path else self._persist_dir
        if target is None:
            return
        fpath = target / "memory.json"
        if not fpath.exists():
            return
        data = json.loads(fpath.read_text())
        self._turn_counter = data.get("turn", 0)
        for k, v in data.get("slots", {}).items():
            self.put(k, v["value"], AgentRole(v["source"]))
        for k, v in data.get("buffers", {}).items():
            self._agent_buffers[AgentRole(k)] = v

    # ── Stats ──

    def stats(self) -> dict[str, Any]:
        return {
            "total_slots": len(self._slots),
            "turn": self._turn_counter,
            "buffer_sizes": {k.value: len(v) for k, v in self._agent_buffers.items()},
        }
