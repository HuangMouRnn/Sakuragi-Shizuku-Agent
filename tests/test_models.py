"""Tests for core data models."""

import json
import pytest
from core.models import (
    AgentMessage,
    AgentRole,
    ArchitectOutput,
    CodeGenOutput,
    DebugOutput,
    GeneratedFile,
    MessageType,
    PipelineState,
    RequirementOutput,
    TestOutput,
    TestResult,
    UserStory,
)


class TestAgentMessage:
    def test_create_message(self):
        msg = AgentMessage(
            sender=AgentRole.REQUIREMENT,
            receiver=AgentRole.ARCHITECT,
            msg_type=MessageType.REQUEST,
            payload={"test": "data"},
        )
        assert msg.sender == AgentRole.REQUIREMENT
        assert msg.receiver == AgentRole.ARCHITECT
        assert len(msg.id) == 12

    def test_reply(self):
        msg = AgentMessage(
            sender=AgentRole.REQUIREMENT,
            receiver=AgentRole.ARCHITECT,
            msg_type=MessageType.REQUEST,
            payload={"req": "build something"},
        )
        reply = msg.reply(AgentRole.ARCHITECT, {"modules": ["a", "b"]})
        assert reply.sender == AgentRole.ARCHITECT
        assert reply.receiver == AgentRole.REQUIREMENT
        assert reply.parent_id == msg.id


class TestRequirementOutput:
    def test_serialization(self):
        output = RequirementOutput(
            project_name="test-proj",
            summary="A test project",
            user_stories=[
                UserStory(id="US-001", role="user", action="do thing", benefit="get value", priority=1)
            ],
            functional_requirements=["FR-1: must work"],
            non_functional_requirements=["NFR-1: fast"],
            tech_constraints=["Python 3.11+"],
            assumptions=["we have a database"],
        )
        data = output.model_dump()
        assert data["project_name"] == "test-proj"
        assert len(data["user_stories"]) == 1

        # Roundtrip
        restored = RequirementOutput(**data)
        assert restored.project_name == output.project_name


class TestArchitectOutput:
    def test_minimal(self):
        output = ArchitectOutput(
            project_name="proj",
            tech_stack={"language": "Python"},
            modules=[],
            directory_structure={"src/": []},
        )
        assert output.project_name == "proj"
        assert output.data_models == []


class TestPipelineState:
    def test_default_state(self):
        state = PipelineState()
        assert state.status == "pending"
        assert state.iteration == 0
        assert state.generated_code == {}

    def test_serialization(self):
        state = PipelineState(status="running", iteration=2)
        data = state.model_dump()
        restored = PipelineState(**data)
        assert restored.status == "running"
        assert restored.iteration == 2
