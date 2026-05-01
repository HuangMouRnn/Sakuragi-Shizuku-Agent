"""Test Agent: generates and runs tests for generated code."""

from __future__ import annotations

import json
import subprocess
import tempfile
import os
from pathlib import Path

from core.base_agent import BaseAgent
from core.models import AgentMessage, AgentRole, TestOutput, TestResult
from prompts.templates import TEST_SYSTEM, TEST_USER


class TestAgent(BaseAgent):
    role = AgentRole.TEST
    name = "TestAgent"

    def system_prompt(self) -> str:
        return TEST_SYSTEM

    async def execute(self, message: AgentMessage) -> AgentMessage:
        payload = message.payload
        code_files = payload.get("code_files", [])
        requirement = payload.get("requirement", {})
        architecture = payload.get("architecture", {})
        previous_failures = payload.get("previous_failures", [])

        # Format code files for prompt
        code_text = ""
        for f in code_files:
            if isinstance(f, dict):
                path = f.get("path", "")
                content = f.get("content", "")
            else:
                path = getattr(f, "path", "")
                content = getattr(f, "content", "")
            code_text += f"\n### {path}\n```python\n{content}\n```\n"

        failures_text = "\n".join(previous_failures) if previous_failures else "None"

        user_prompt = TEST_USER.format(
            code_files=code_text,
            requirement_json=json.dumps(requirement, indent=2),
            architecture_json=json.dumps(architecture, indent=2),
            previous_failures=failures_text,
        )

        result = await self._llm_generate_json(user_prompt, temperature=0.1, max_tokens=8192)

        # Try to actually run the generated tests
        test_cases = result.get("test_cases", [])
        if test_cases:
            results = await self._run_tests(test_cases, code_files)
            result["results"] = results
            result["all_passed"] = all(r.get("passed", False) for r in results)
            failures = [r for r in results if not r.get("passed", False)]
            result["failure_summary"] = (
                f"{len(failures)} test(s) failed:\n" +
                "\n".join(f"- {r.get('test_name')}: {r.get('error', '')[:200]}" for r in failures)
            ) if failures else ""

        try:
            output = TestOutput(**result)
        except Exception:
            result.setdefault("test_cases", [])
            result.setdefault("results", [])
            result.setdefault("coverage_pct", 0)
            result.setdefault("all_passed", False)
            result.setdefault("failure_summary", "")
            output = TestOutput(**result)

        self._record(message)
        return message.reply(self.role, output.model_dump())

    async def _run_tests(self, test_cases: list[dict], code_files: list) -> list[dict]:
        """Actually execute test cases in a temporary environment."""
        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write source files
            for f in code_files:
                if isinstance(f, dict):
                    path, content = f.get("path", ""), f.get("content", "")
                else:
                    path, content = getattr(f, "path", ""), getattr(f, "content", "")
                fpath = Path(tmpdir) / path
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)

                # Create __init__.py if needed
                init_path = fpath.parent / "__init__.py"
                if not init_path.exists():
                    init_path.write_text("")

            # Write and run each test
            for tc in test_cases:
                test_name = tc.get("name", "unknown")
                test_code = tc.get("test_code", "")
                test_file = Path(tmpdir) / f"test_{test_name}.py"
                test_file.write_text(test_code)

                try:
                    proc = subprocess.run(
                        ["python", "-m", "pytest", str(test_file), "-x", "-v", "--tb=short"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=tmpdir,
                        env={**os.environ, "PYTHONPATH": tmpdir},
                    )
                    results.append({
                        "test_name": test_name,
                        "passed": proc.returncode == 0,
                        "output": proc.stdout[-2000:] if proc.stdout else "",
                        "error": proc.stderr[-2000:] if proc.returncode != 0 else None,
                        "duration_ms": 0,
                    })
                except subprocess.TimeoutExpired:
                    results.append({
                        "test_name": test_name,
                        "passed": False,
                        "output": "",
                        "error": "Test timed out after 30 seconds",
                        "duration_ms": 30000,
                    })
                except Exception as e:
                    results.append({
                        "test_name": test_name,
                        "passed": False,
                        "output": "",
                        "error": str(e),
                        "duration_ms": 0,
                    })

        return results
