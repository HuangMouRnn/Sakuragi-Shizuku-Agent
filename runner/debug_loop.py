"""Self-healing debug loop: orchestrates the test → fix → retest cycle."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.base_agent import BaseAgent, LLMClient
from core.memory_manager import MemoryManager
from core.models import (
    AgentMessage,
    AgentRole,
    ArchitectOutput,
    DebugInput,
    DebugOutput,
    GeneratedFile,
    MessageType,
    RequirementOutput,
    TestInput,
    TestOutput,
    TestResult,
)

logger = logging.getLogger(__name__)


@dataclass
class DebugLoopConfig:
    max_iterations: int = 3
    backoff_factor: float = 1.5  # increase temperature on each retry
    base_temperature: float = 0.2
    max_temperature: float = 0.7
    save_intermediate: bool = True
    output_dir: str = "./debug_output"


@dataclass
class DebugLoopResult:
    success: bool
    final_code: dict[str, str]
    iterations: int
    history: list[dict[str, Any]] = field(default_factory=list)
    final_test_output: TestOutput | None = None


class DebugLoop:
    """
    Implements the self-healing loop:
    1. Generate test
    2. Run test
    3. If fail → analyze → patch → goto 1
    4. If pass → done
    """

    def __init__(
        self,
        test_agent: BaseAgent,
        debug_agent: BaseAgent,
        code_gen_agent: BaseAgent,
        config: DebugLoopConfig | None = None,
        memory: MemoryManager | None = None,
        on_event: Callable | None = None,
    ):
        self.test_agent = test_agent
        self.debug_agent = debug_agent
        self.code_gen_agent = code_gen_agent
        self.config = config or DebugLoopConfig()
        self.memory = memory or MemoryManager()
        self._on_event = on_event

    async def run(
        self,
        code_files: list[GeneratedFile],
        requirement: RequirementOutput,
        architecture: ArchitectOutput,
    ) -> DebugLoopResult:
        """Execute the full debug loop for a set of code files."""
        current_code = {f.path: f.content for f in code_files}
        history: list[dict[str, Any]] = []

        for iteration in range(self.config.max_iterations):
            logger.info("Debug loop iteration %d/%d", iteration + 1, self.config.max_iterations)

            # Step 1: Run tests
            test_output = await self._run_tests(current_code, requirement, architecture)

            iteration_record = {
                "iteration": iteration,
                "tests_total": len(test_output.test_cases),
                "tests_passed": sum(1 for r in test_output.results if r.passed),
                "tests_failed": sum(1 for r in test_output.results if not r.passed),
                "all_passed": test_output.all_passed,
            }

            if test_output.all_passed:
                logger.info("All tests passed at iteration %d", iteration)
                iteration_record["action"] = "passed"
                history.append(iteration_record)
                return DebugLoopResult(
                    success=True,
                    final_code=current_code,
                    iterations=iteration + 1,
                    history=history,
                    final_test_output=test_output,
                )

            # Step 2: Analyze failures
            failed_tests = [r for r in test_output.results if not r.passed]
            iteration_record["failure_summary"] = test_output.failure_summary

            await self._emit("test_failed", {
                "iteration": iteration,
                "failures": len(failed_tests),
                "summary": test_output.failure_summary[:500],
            })

            # Step 3: Debug - get patches
            temperature = min(
                self.config.base_temperature + iteration * self.config.backoff_factor * 0.1,
                self.config.max_temperature,
            )

            debug_output = await self._analyze_failures(
                failed_tests=failed_tests,
                source_code=current_code,
                test_code={tc.name: tc.test_code for tc in test_output.test_cases},
                architecture=architecture,
                attempt=iteration + 1,
                temperature=temperature,
            )

            iteration_record["root_cause"] = debug_output.root_cause
            iteration_record["patches_count"] = len(debug_output.patches)
            iteration_record["confidence"] = debug_output.confidence

            # Step 4: Apply patches
            if not debug_output.patches:
                logger.warning("No patches produced at iteration %d", iteration)
                iteration_record["action"] = "no_patches"
                history.append(iteration_record)

                # Try full regeneration as last resort
                if iteration == self.config.max_iterations - 1:
                    current_code = await self._regenerate_module(
                        architecture, requirement, current_code, debug_output.root_cause
                    )
                continue

            applied = 0
            for patch in debug_output.patches:
                if patch.file_path in current_code:
                    if patch.original_snippet in current_code[patch.file_path]:
                        current_code[patch.file_path] = current_code[patch.file_path].replace(
                            patch.original_snippet, patch.fixed_snippet, 1
                        )
                        applied += 1
                    else:
                        logger.warning("Patch target not found in %s", patch.file_path)

            iteration_record["patches_applied"] = applied
            iteration_record["action"] = "patched"

            await self._emit("patches_applied", {
                "iteration": iteration,
                "applied": applied,
                "root_cause": debug_output.root_cause,
            })

            # Save intermediate state
            if self.config.save_intermediate:
                self._save_snapshot(current_code, iteration)

            history.append(iteration_record)

            self.memory.append_log(AgentRole.DEBUG, iteration_record)

        # Max iterations reached
        logger.warning("Debug loop exhausted %d iterations", self.config.max_iterations)
        final_test = await self._run_tests(current_code, requirement, architecture)

        return DebugLoopResult(
            success=final_test.all_passed,
            final_code=current_code,
            iterations=self.config.max_iterations,
            history=history,
            final_test_output=final_test,
        )

    async def _run_tests(
        self,
        code: dict[str, str],
        req: RequirementOutput,
        arch: ArchitectOutput,
    ) -> TestOutput:
        files = [GeneratedFile(path=p, content=c) for p, c in code.items()]
        msg = AgentMessage(
            sender=AgentRole.TEST,
            receiver=AgentRole.TEST,
            msg_type=MessageType.REQUEST,
            payload=TestInput(code_files=files, requirement=req, architecture=arch).model_dump(),
        )
        result = await self.test_agent.execute(msg)
        return TestOutput(**result.payload)

    async def _analyze_failures(
        self,
        failed_tests: list[TestResult],
        source_code: dict[str, str],
        test_code: dict[str, str],
        architecture: ArchitectOutput,
        attempt: int,
        temperature: float,
    ) -> DebugOutput:
        msg = AgentMessage(
            sender=AgentRole.DEBUG,
            receiver=AgentRole.DEBUG,
            msg_type=MessageType.REQUEST,
            payload=DebugInput(
                failed_tests=failed_tests,
                source_code=source_code,
                test_code=test_code,
                architecture=architecture,
                attempt_number=attempt,
                max_attempts=self.config.max_iterations,
            ).model_dump(),
        )
        # Override temperature for this call
        original_generate = self.debug_agent._llm_generate_json

        async def temp_generate(prompt, **kwargs):
            kwargs["temperature"] = temperature
            return await original_generate(prompt, **kwargs)

        self.debug_agent._llm_generate_json = temp_generate
        try:
            result = await self.debug_agent.execute(msg)
        finally:
            self.debug_agent._llm_generate_json = original_generate

        return DebugOutput(**result.payload)

    async def _regenerate_module(
        self,
        arch: ArchitectOutput,
        req: RequirementOutput,
        current_code: dict[str, str],
        error_context: str,
    ) -> dict[str, str]:
        """Last resort: regenerate the problematic module from scratch."""
        logger.info("Attempting full module regeneration")
        # Find the first module that likely has issues
        for module in arch.modules:
            msg = AgentMessage(
                sender=AgentRole.CODE_GENERATOR,
                receiver=AgentRole.CODE_GENERATOR,
                msg_type=MessageType.REQUEST,
                payload={
                    "architecture": arch.model_dump(),
                    "module_name": module.name,
                    "previous_code": {},
                    "error_feedback": f"Previous attempt failed: {error_context}. Regenerate completely.",
                },
            )
            result = await self.code_gen_agent.execute(msg)
            from core.models import CodeGenOutput
            gen = CodeGenOutput(**result.payload)
            for f in gen.files:
                current_code[f.path] = f.content

        return current_code

    def _save_snapshot(self, code: dict[str, str], iteration: int):
        out_dir = Path(self.config.output_dir) / "snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_file = out_dir / f"iteration_{iteration}.json"
        snap_file.write_text(json.dumps(code, indent=2))

    async def _emit(self, event: str, data: dict):
        if self._on_event:
            if asyncio.iscoroutinefunction(self._on_event):
                await self._on_event(event, data)
            else:
                self._on_event(event, data)
