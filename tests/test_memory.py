"""Tests for memory manager and context optimizer."""

import json
import pytest
from core.memory_manager import MemoryManager, MemorySlot
from core.context_optimizer import ContextOptimizer
from core.models import AgentRole, PipelineState


class TestMemorySlot:
    def test_access_count(self):
        slot = MemorySlot("key", "value", AgentRole.DEBUG)
        assert slot.access_count == 0
        slot.access()
        slot.access()
        assert slot.access_count == 2

    def test_expiration(self):
        slot = MemorySlot("key", "value", AgentRole.DEBUG, ttl=3)
        assert not slot.is_expired(0)
        slot.access()
        slot.access()
        assert not slot.is_expired(0)
        slot.access()
        assert slot.is_expired(0)


class TestMemoryManager:
    def test_put_get(self):
        mm = MemoryManager()
        mm.put("test_key", {"data": 42}, AgentRole.ARCHITECT)
        result = mm.get("test_key")
        assert result == {"data": 42}

    def test_get_missing(self):
        mm = MemoryManager()
        assert mm.get("nonexistent") is None
        assert mm.get("nonexistent", "default") == "default"

    def test_get_prefix(self):
        mm = MemoryManager()
        mm.put("code/module_a", "a", AgentRole.CODE_GENERATOR)
        mm.put("code/module_b", "b", AgentRole.CODE_GENERATOR)
        mm.put("req/main", "r", AgentRole.REQUIREMENT)
        result = mm.get_prefix("code/")
        assert len(result) == 2

    def test_eviction(self):
        mm = MemoryManager(max_slots=5)
        for i in range(10):
            mm.put(f"key_{i}", f"value_{i}", AgentRole.DEBUG)
        assert len(mm._slots) <= 5

    def test_agent_buffers(self):
        mm = MemoryManager()
        mm.append_log(AgentRole.DEBUG, {"action": "patch", "file": "test.py"})
        mm.append_log(AgentRole.DEBUG, {"action": "retest", "result": "pass"})
        logs = mm.get_logs(AgentRole.DEBUG)
        assert len(logs) == 2
        assert logs[0]["action"] == "patch"

    def test_agent_buffers_last_n(self):
        mm = MemoryManager()
        for i in range(10):
            mm.append_log(AgentRole.TEST, {"i": i})
        assert len(mm.get_logs(AgentRole.TEST, last_n=3)) == 3

    def test_pipeline_state(self):
        mm = MemoryManager()
        state = PipelineState(status="running")
        mm.save_state(state)
        loaded = mm.load_state()
        assert loaded is not None
        assert loaded.status == "running"

    def test_advance_turn(self):
        mm = MemoryManager()
        mm.put("temp", "value", AgentRole.DEBUG, ttl=2)
        mm.advance_turn()
        assert mm.get("temp") == "value"
        mm.advance_turn()
        # ttl=2 means expired after 2 accesses, not 2 turns
        # But with is_expired checking access_count >= ttl

    def test_stats(self):
        mm = MemoryManager()
        mm.put("a", 1, AgentRole.DEBUG)
        mm.put("b", 2, AgentRole.TEST)
        stats = mm.stats()
        assert stats["total_slots"] == 2
        assert stats["turn"] == 0

    def test_persistence(self, tmp_path):
        mm = MemoryManager(persist_dir=str(tmp_path))
        mm.put("saved", {"data": True}, AgentRole.ARCHITECT)
        mm.save_to_disk()

        mm2 = MemoryManager(persist_dir=str(tmp_path))
        mm2.load_from_disk()
        assert mm2.get("saved") is not None


class TestContextOptimizer:
    def test_count_tokens(self):
        opt = ContextOptimizer()
        count = opt.count_tokens("Hello, world!")
        assert count > 0
        assert count < 10

    def test_truncate(self):
        opt = ContextOptimizer()
        long_text = "word " * 1000
        truncated = opt.truncate_to_tokens(long_text, 50)
        assert len(truncated) < len(long_text)
        assert truncated.endswith("[truncated]")

    def test_compress_code(self):
        code = '''def hello():
    """This is a docstring."""
    # This is a comment
    return "world"

def other():
    """Another docstring."""
    return 42
'''
        compressed = ContextOptimizer.compress_code(code)
        assert "docstring" not in compressed
        assert "def hello" in compressed
        assert "def other" in compressed

    def test_summarize_structure(self):
        code = '''import os

class MyClass:
    """A class."""
    def method(self):
        pass

def standalone():
    pass
'''
        structure = ContextOptimizer.summarize_code_structure(code)
        assert "class MyClass" in structure
        assert "def standalone" in structure
        assert "import os" not in structure

    def test_chunk_codebase(self):
        opt = ContextOptimizer()
        files = {f"file_{i}.py": f"# Content of file {i}\n" * 50 for i in range(20)}
        chunks = opt.chunk_codebase(files, chunk_tokens=500)
        assert len(chunks) >= 1
        # Each chunk should be a dict
        for chunk in chunks:
            assert isinstance(chunk, dict)
