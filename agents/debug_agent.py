"""Debug Agent: analyzes test failures and produces precise code patches."""

from __future__ import annotations

import json

from core.base_agent import BaseAgent
from core.context_optimizer import ContextOptimizer
from core.models import AgentMessage, AgentRole, DebugOutput
from prompts.templates import DEBUG_SYSTEM, DEBUG_USER


class DebugAgent(BaseAgent):
    role = AgentRole.DEBUG
    name = "DebugAgent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = ContextOptimizer()

    def system_prompt(self) -> str:
        return DEBUG_SYSTEM

    async def execute(self, message: AgentMessage) -> AgentMessage:
        payload = message.payload
        failed_tests = payload.get("failed_tests", [])
        source_code = payload.get("source_code", {})
        test_code = payload.get("test_code", {})
        architecture = payload.get("architecture", {})
        attempt_number = payload.get("attempt_number", 1)
        max_attempts = payload.get("max_attempts", 3)

        # Format failed tests
        failed_text = ""
        for ft in failed_tests:
            if isinstance(ft, dict):
                failed_text += f"""
### {ft.get('test_name', 'unknown')}
- Passed: {ft.get('passed', False)}
- Error: {ft.get('error', 'N/A')[:500]}
- Output: {ft.get('output', '')[:300]}
"""
            else:
                failed_text += f"""
### {getattr(ft, 'test_name', 'unknown')}
- Passed: {getattr(ft, 'passed', False)}
- Error: {str(getattr(ft, 'error', 'N/A'))[:500]}
"""

        # Compress source code if too large
        source_text = ""
        for path, content in source_code.items():
            if len(content) > 3000:
                compressed = ContextOptimizer.compress_code(content)
                source_text += f"\n### {path}\n```python\n{compressed}\n```\n"
            else:
                source_text += f"\n### {path}\n```python\n{content}\n```\n"

        test_text = ""
        for name, code in test_code.items():
            test_text += f"\n### {name}\n```python\n{code[:2000]}\n```\n"

        user_prompt = DEBUG_USER.format(
            failed_tests=failed_text,
            source_code=source_text,
            test_code=test_text,
            architecture_json=json.dumps(architecture, indent=2)[:3000],
            attempt_number=attempt_number,
            max_attempts=max_attempts,
        )

        # Optimize context
        sys_prompt, user_prompt = self.optimizer.optimize_context(
            system_prompt=self.system_prompt(),
            user_prompt=user_prompt,
            context_parts={
                "source_code": source_text,
                "test_code": test_text,
                "failed_tests": failed_text,
            },
            role="debug",
        )

        result = await self._llm_generate_json(user_prompt, temperature=0.3, max_tokens=4096)

        try:
            output = DebugOutput(**result)
        except Exception:
            result.setdefault("root_cause", "Unknown")
            result.setdefault("patches", [])
            result.setdefault("reasoning_chain", [])
            result.setdefault("confidence", 0.5)
            result.setdefault("needs_new_tests", False)
            output = DebugOutput(**result)

        self._record(message)
        return message.reply(self.role, output.model_dump())
