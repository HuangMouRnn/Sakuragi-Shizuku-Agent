"""Agent orchestrator: manages agent lifecycle, message routing, and execution flow."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .base_agent import BaseAgent, LLMClient
from .context_optimizer import ContextOptimizer
from .memory_manager import MemoryManager
from .models import (
    AgentMessage,
    AgentRole,
    ArchitectInput,
    ArchitectOutput,
    CodeGenInput,
    CodeGenOutput,
    DebugInput,
    DebugOutput,
    MessageType,
    PipelineState,
    RequirementInput,
    RequirementOutput,
    TestInput,
    TestOutput,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestrator that drives the multi-agent pipeline.

    Flow: Requirement → Architect → [CodeGen → Test → Debug]* per module
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        memory: MemoryManager | None = None,
        output_dir: str = "./output",
        max_debug_iterations: int = 3,
        verbose: bool = False,
    ):
        self.llm = llm or LLMClient()
        self.memory = memory or MemoryManager()
        self.optimizer = ContextOptimizer(model=self.llm.model)
        self.output_dir = Path(output_dir)
        self.max_debug_iterations = max_debug_iterations
        self.verbose = verbose

        self._agents: dict[AgentRole, BaseAgent] = {}
        self._state = PipelineState()
        self._callbacks: list[Any] = []

    # ── Agent Registration ──

    def register(self, agent: BaseAgent):
        self._agents[agent.role] = agent
        logger.info("Registered agent: %s (%s)", agent.name, agent.role.value)

    def _get_agent(self, role: AgentRole) -> BaseAgent:
        if role not in self._agents:
            raise ValueError(f"Agent not registered: {role.value}")
        return self._agents[role]

    # ── Callbacks ──

    def on(self, callback):
        self._callbacks.append(callback)

    async def _emit(self, event: str, data: dict):
        for cb in self._callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(event, data)
            else:
                cb(event, data)

    # ── Main Pipeline ──

    async def run(self, requirement_text: str, project_name: str = "auto-project") -> PipelineState:
        """Execute the full development pipeline."""
        self._state = PipelineState(status="running")
        self._state.max_iterations = self.max_debug_iterations
        await self._emit("pipeline_start", {"requirement": requirement_text})

        try:
            # Phase 1: Requirement Analysis
            await self._emit("phase_start", {"phase": "requirement"})
            req_output = await self._run_requirement(requirement_text, project_name)
            self._state.requirement = req_output
            self.memory.put("requirement", req_output.model_dump(), AgentRole.REQUIREMENT)
            await self._emit("phase_complete", {"phase": "requirement", "output": req_output.model_dump()})

            # Phase 2: Architecture Design
            await self._emit("phase_start", {"phase": "architect"})
            arch_output = await self._run_architect(req_output)
            self._state.architecture = arch_output
            self._state.total_modules = len(arch_output.modules)
            self.memory.put("architecture", arch_output.model_dump(), AgentRole.ARCHITECT)
            await self._emit("phase_complete", {"phase": "architect", "output": arch_output.model_dump()})

            # Phase 3: Iterative Code Generation + Testing + Debugging per module
            all_generated: dict[str, str] = {}

            for idx, module in enumerate(arch_output.modules):
                self._state.current_module_idx = idx
                await self._emit("module_start", {"module": module.name, "index": idx, "total": len(arch_output.modules)})

                module_code = await self._run_module_pipeline(module.name, arch_output, req_output, all_generated)
                all_generated.update(module_code)

                self._state.generated_code = all_generated
                await self._emit("module_complete", {"module": module.name, "files": list(module_code.keys())})

            # Phase 4: Final Integration Test
            await self._emit("phase_start", {"phase": "integration_test"})
            final_test = await self._run_integration_test(all_generated, req_output, arch_output)

            if final_test.all_passed:
                self._state.status = "success"
            else:
                self._state.status = "failed"
                self._state.test_output = final_test

            # Write output files
            self._write_project(all_generated, arch_output)

            await self._emit("pipeline_complete", {"status": self._state.status})
            return self._state

        except Exception as e:
            logger.exception("Pipeline failed")
            self._state.status = "failed"
            await self._emit("pipeline_error", {"error": str(e)})
            return self._state

    # ── Phase Runners ──

    async def _run_requirement(self, text: str, project_name: str) -> RequirementOutput:
        agent = self._get_agent(AgentRole.REQUIREMENT)
        msg = AgentMessage(
            sender=AgentRole.REQUIREMENT,
            receiver=AgentRole.REQUIREMENT,
            msg_type=MessageType.REQUEST,
            payload=RequirementInput(raw_requirement=text, project_name=project_name).model_dump(),
        )
        result = await agent.execute(msg)
        return RequirementOutput(**result.payload)

    async def _run_architect(self, req: RequirementOutput) -> ArchitectOutput:
        agent = self._get_agent(AgentRole.ARCHITECT)
        msg = AgentMessage(
            sender=AgentRole.ARCHITECT,
            receiver=AgentRole.ARCHITECT,
            msg_type=MessageType.REQUEST,
            payload=ArchitectInput(requirement=req).model_dump(),
        )
        result = await agent.execute(msg)
        return ArchitectOutput(**result.payload)

    async def _run_module_pipeline(
        self,
        module_name: str,
        arch: ArchitectOutput,
        req: RequirementOutput,
        existing_code: dict[str, str],
    ) -> dict[str, str]:
        """Run code-gen → test → debug loop for a single module."""
        gen_agent = self._get_agent(AgentRole.CODE_GENERATOR)
        test_agent = self._get_agent(AgentRole.TEST)
        debug_agent = self._get_agent(AgentRole.DEBUG)

        current_code: dict[str, str] = dict(existing_code)
        error_feedback: str | None = None

        for iteration in range(self.max_debug_iterations + 1):
            self._state.iteration = iteration
            self.memory.advance_turn()

            # Generate code
            gen_msg = AgentMessage(
                sender=AgentRole.CODE_GENERATOR,
                receiver=AgentRole.CODE_GENERATOR,
                msg_type=MessageType.REQUEST,
                payload=CodeGenInput(
                    architecture=arch,
                    module_name=module_name,
                    previous_code=current_code,
                    error_feedback=error_feedback,
                ).model_dump(),
            )
            gen_result = await gen_agent.execute(gen_msg)
            gen_output = CodeGenOutput(**gen_result.payload)

            # Merge new code
            for f in gen_output.files:
                current_code[f.path] = f.content
            error_feedback = None

            await self._emit("code_generated", {
                "module": module_name,
                "iteration": iteration,
                "files": [f.path for f in gen_output.files],
            })

            # Run tests
            test_msg = AgentMessage(
                sender=AgentRole.TEST,
                receiver=AgentRole.TEST,
                msg_type=MessageType.REQUEST,
                payload=TestInput(
                    code_files=gen_output.files,
                    requirement=req,
                    architecture=arch,
                ).model_dump(),
            )
            test_result = await test_agent.execute(test_msg)
            test_output = TestOutput(**test_result.payload)

            await self._emit("test_complete", {
                "module": module_name,
                "iteration": iteration,
                "passed": test_output.all_passed,
                "failures": test_output.failure_summary,
            })

            if test_output.all_passed:
                logger.info("Module '%s' passed all tests (iteration %d)", module_name, iteration)
                return {f.path: f.content for f in gen_output.files}

            # Debug and fix
            if iteration >= self.max_debug_iterations:
                logger.warning("Module '%s' failed after %d iterations", module_name, iteration)
                return current_code

            debug_msg = AgentMessage(
                sender=AgentRole.DEBUG,
                receiver=AgentRole.DEBUG,
                msg_type=MessageType.REQUEST,
                payload=DebugInput(
                    failed_tests=[r for r in test_output.results if not r.passed],
                    source_code=current_code,
                    test_code={tc.name: tc.test_code for tc in test_output.test_cases},
                    architecture=arch,
                    attempt_number=iteration + 1,
                    max_attempts=self.max_debug_iterations,
                ).model_dump(),
            )
            debug_result = await debug_agent.execute(debug_msg)
            debug_output = DebugOutput(**debug_result.payload)

            self._state.debug_history.append(debug_output)
            self.memory.append_log(AgentRole.DEBUG, {
                "module": module_name,
                "root_cause": debug_output.root_cause,
                "patches": len(debug_output.patches),
            })

            # Apply patches
            for patch in debug_output.patches:
                if patch.file_path in current_code:
                    current_code[patch.file_path] = current_code[patch.file_path].replace(
                        patch.original_snippet, patch.fixed_snippet
                    )

            error_feedback = f"Root cause: {debug_output.root_cause}\nPatches applied: {len(debug_output.patches)}"
            await self._emit("debug_applied", {
                "module": module_name,
                "iteration": iteration,
                "root_cause": debug_output.root_cause,
                "patches": len(debug_output.patches),
            })

        return current_code

    async def _run_integration_test(
        self,
        code: dict[str, str],
        req: RequirementOutput,
        arch: ArchitectOutput,
    ) -> TestOutput:
        test_agent = self._get_agent(AgentRole.TEST)
        from .models import GeneratedFile
        files = [GeneratedFile(path=p, content=c) for p, c in code.items()]
        msg = AgentMessage(
            sender=AgentRole.TEST,
            receiver=AgentRole.TEST,
            msg_type=MessageType.REQUEST,
            payload=TestInput(code_files=files, requirement=req, architecture=arch).model_dump(),
        )
        result = await test_agent.execute(msg)
        return TestOutput(**result.payload)

    # ── Output ──

    def _write_project(self, code: dict[str, str], arch: ArchitectOutput):
        project_dir = self.output_dir / arch.project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        for path, content in code.items():
            fpath = project_dir / path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)
        logger.info("Project written to %s (%d files)", project_dir, len(code))
