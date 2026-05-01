"""Data models for inter-agent communication."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    REQUIREMENT = "requirement"
    ARCHITECT = "architect"
    CODE_GENERATOR = "code_generator"
    TEST = "test"
    DEBUG = "debug"


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    FEEDBACK = "feedback"


class AgentMessage(BaseModel):
    """Structured message passed between agents."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: AgentRole
    receiver: AgentRole
    msg_type: MessageType
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    parent_id: str | None = None

    def reply(self, sender: AgentRole, payload: dict[str, Any], msg_type: MessageType = MessageType.RESPONSE) -> AgentMessage:
        return AgentMessage(
            sender=sender,
            receiver=self.sender,
            msg_type=msg_type,
            payload=payload,
            parent_id=self.id,
        )


# ── Requirement Agent I/O ──

class RequirementInput(BaseModel):
    raw_requirement: str
    project_name: str = "untitled"
    constraints: list[str] = Field(default_factory=list)


class UserStory(BaseModel):
    id: str
    role: str
    action: str
    benefit: str
    priority: int = 1


class RequirementOutput(BaseModel):
    project_name: str
    summary: str
    user_stories: list[UserStory]
    functional_requirements: list[str]
    non_functional_requirements: list[str]
    tech_constraints: list[str]
    assumptions: list[str]


# ── Architect Agent I/O ──

class ArchitectInput(BaseModel):
    requirement: RequirementOutput
    preferred_stack: list[str] = Field(default_factory=list)


class ModuleSpec(BaseModel):
    name: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    public_api: list[str] = Field(default_factory=list)


class ArchitectOutput(BaseModel):
    project_name: str
    tech_stack: dict[str, str]
    modules: list[ModuleSpec]
    directory_structure: dict[str, Any]
    data_models: list[dict[str, Any]] = Field(default_factory=list)
    api_endpoints: list[dict[str, Any]] = Field(default_factory=list)
    sequence_diagram: str = ""


# ── Code Generator Agent I/O ──

class CodeGenInput(BaseModel):
    architecture: ArchitectOutput
    module_name: str
    context_modules: list[str] = Field(default_factory=list)
    previous_code: dict[str, str] = Field(default_factory=dict)
    error_feedback: str | None = None


class GeneratedFile(BaseModel):
    path: str
    content: str
    language: str = "python"


class CodeGenOutput(BaseModel):
    module_name: str
    files: list[GeneratedFile]
    dependencies: list[str] = Field(default_factory=list)
    notes: str = ""


# ── Test Agent I/O ──

class TestInput(BaseModel):
    code_files: list[GeneratedFile]
    requirement: RequirementOutput
    architecture: ArchitectOutput
    previous_failures: list[str] = Field(default_factory=list)


class TestCase(BaseModel):
    name: str
    description: str
    test_code: str
    target_file: str
    test_type: str = "unit"  # unit / integration / e2e


class TestResult(BaseModel):
    test_name: str
    passed: bool
    output: str
    error: str | None = None
    duration_ms: float = 0


class TestOutput(BaseModel):
    test_cases: list[TestCase]
    results: list[TestResult] = Field(default_factory=list)
    coverage_pct: float = 0
    all_passed: bool = False
    failure_summary: str = ""


# ── Debug Agent I/O ──

class DebugInput(BaseModel):
    failed_tests: list[TestResult]
    source_code: dict[str, str]  # path -> content
    test_code: dict[str, str]
    architecture: ArchitectOutput
    attempt_number: int = 1
    max_attempts: int = 3


class FixPatch(BaseModel):
    file_path: str
    original_snippet: str
    fixed_snippet: str
    explanation: str


class DebugOutput(BaseModel):
    root_cause: str
    patches: list[FixPatch]
    reasoning_chain: list[str]
    confidence: float = 0.0
    needs_new_tests: bool = False


# ── Pipeline State ──

class PipelineState(BaseModel):
    """Tracks the full state of a development pipeline run."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    requirement: RequirementOutput | None = None
    architecture: ArchitectOutput | None = None
    generated_code: dict[str, str] = Field(default_factory=dict)
    test_output: TestOutput | None = None
    debug_history: list[DebugOutput] = Field(default_factory=list)
    current_module_idx: int = 0
    total_modules: int = 0
    iteration: int = 0
    max_iterations: int = 5
    status: str = "pending"  # pending / running / success / failed
