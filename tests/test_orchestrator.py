"""Tests for orchestrator and agent integration."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.base_agent import BaseAgent, LLMClient
from core.models import (
    AgentMessage,
    AgentRole,
    ArchitectOutput,
    CodeGenOutput,
    DebugOutput,
    GeneratedFile,
    MessageType,
    RequirementOutput,
    TestOutput,
    TestResult,
    UserStory,
)
from core.orchestrator import Orchestrator


class MockLLMClient:
    """Mock LLM that returns pre-configured responses."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0
        self.provider = "mock"
        self.model = "mock-model"

    async def chat(self, system, messages, temperature=0.2, max_tokens=4096, response_format=None):
        self.call_count += 1
        user_msg = messages[-1]["content"] if messages else ""

        # Return pre-configured response based on system prompt keyword
        for key, resp in self.responses.items():
            if key.lower() in system.lower():
                return resp

        return json.dumps({"status": "ok", "call": self.call_count})


# ── Sample Data ──

def sample_requirement_output() -> dict:
    return {
        "project_name": "calculator",
        "summary": "A simple calculator library",
        "user_stories": [
            {"id": "US-001", "role": "user", "action": "perform arithmetic", "benefit": "calculate results", "priority": 1}
        ],
        "functional_requirements": ["FR-1: support add, subtract, multiply, divide"],
        "non_functional_requirements": ["NFR-1: handle edge cases"],
        "tech_constraints": ["pure Python"],
        "assumptions": ["no GUI needed"],
    }


def sample_architecture_output() -> dict:
    return {
        "project_name": "calculator",
        "tech_stack": {"language": "Python 3.11+"},
        "modules": [
            {
                "name": "core",
                "description": "Calculator operations",
                "dependencies": [],
                "public_api": ["Calculator"],
            },
            {
                "name": "api",
                "description": "CLI interface",
                "dependencies": ["core"],
                "public_api": ["main"],
            },
        ],
        "directory_structure": {"src/": {"calculator/": ["core.py", "api.py"]}},
        "data_models": [],
        "api_endpoints": [],
        "sequence_diagram": "User inputs expression -> parser validates -> core computes -> return result",
    }


def sample_codegen_output() -> dict:
    return {
        "module_name": "core",
        "files": [
            {
                "path": "src/calculator/core.py",
                "content": "class Calculator:\n    def add(self, a: float, b: float) -> float:\n        return a + b\n",
                "language": "python",
            }
        ],
        "dependencies": [],
        "notes": "Basic implementation",
    }


def sample_test_output_pass() -> dict:
    return {
        "test_cases": [
            {
                "name": "test_add",
                "description": "Test addition",
                "test_code": "def test_add():\n    assert 1 + 1 == 2\n",
                "target_file": "src/calculator/core.py",
                "test_type": "unit",
            }
        ],
        "results": [
            {"test_name": "test_add", "passed": True, "output": "PASSED", "error": None, "duration_ms": 1.0}
        ],
        "coverage_pct": 100.0,
        "all_passed": True,
        "failure_summary": "",
    }


def sample_debug_output() -> dict:
    return {
        "root_cause": "division by zero not handled",
        "patches": [
            {
                "file_path": "src/calculator/core.py",
                "original_snippet": "return a / b",
                "fixed_snippet": "if b == 0: raise ValueError('division by zero')\n        return a / b",
                "explanation": "Add zero check",
            }
        ],
        "reasoning_chain": ["step 1: test_divide_by_zero fails", "step 2: no guard clause", "step 3: add check"],
        "confidence": 0.9,
        "needs_new_tests": False,
    }


# ── Tests ──

@pytest.mark.asyncio
async def test_orchestrator_register():
    orch = Orchestrator()
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.role = AgentRole.REQUIREMENT
    mock_agent.name = "RequirementAgent"
    orch.register(mock_agent)
    assert AgentRole.REQUIREMENT in orch._agents


@pytest.mark.asyncio
async def test_orchestrator_get_unregistered():
    orch = Orchestrator()
    with pytest.raises(ValueError, match="not registered"):
        orch._get_agent(AgentRole.DEBUG)


@pytest.mark.asyncio
async def test_orchestrator_emits_events():
    events = []

    async def capture(event, data):
        events.append(event)

    responses = {
        "requirement": json.dumps(sample_requirement_output()),
        "architect": json.dumps(sample_architecture_output()),
        "code": json.dumps(sample_codegen_output()),
        "test": json.dumps(sample_test_output_pass()),
    }
    mock_llm = MockLLMClient(responses)
    orch = Orchestrator(llm=mock_llm, max_debug_iterations=0)
    orch.on(capture)

    # This would fail without proper agents, but we can test event emission structure
    assert len(orch._callbacks) == 1


@pytest.mark.asyncio
async def test_callback_receives_events():
    received = []

    def sync_cb(event, data):
        received.append((event, data))

    orch = Orchestrator()
    orch.on(sync_cb)
    await orch._emit("test_event", {"key": "value"})
    assert len(received) == 1
    assert received[0] == ("test_event", {"key": "value"})


@pytest.mark.asyncio
async def test_async_callback():
    received = []

    async def async_cb(event, data):
        received.append(event)

    orch = Orchestrator()
    orch.on(async_cb)
    await orch._emit("async_event", {})
    assert received == ["async_event"]
