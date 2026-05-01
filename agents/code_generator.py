"""Code Generator Agent: produces module code from architecture specs."""

from __future__ import annotations

import json

from core.base_agent import BaseAgent
from core.context_optimizer import ContextOptimizer
from core.models import AgentMessage, AgentRole, CodeGenOutput
from prompts.templates import CODEGEN_ERROR_SECTION, CODEGEN_SYSTEM, CODEGEN_USER


class CodeGeneratorAgent(BaseAgent):
    role = AgentRole.CODE_GENERATOR
    name = "CodeGeneratorAgent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = ContextOptimizer()

    def system_prompt(self) -> str:
        return CODEGEN_SYSTEM

    async def execute(self, message: AgentMessage) -> AgentMessage:
        payload = message.payload
        module_name = payload.get("module_name", "")
        architecture = payload.get("architecture", {})
        previous_code = payload.get("previous_code", {})
        error_feedback = payload.get("error_feedback")

        # Compress previous code to save context
        if previous_code:
            compressed = {}
            for path, content in previous_code.items():
                if len(content) > 2000:
                    compressed[path] = ContextOptimizer.compress_code(content)
                else:
                    compressed[path] = content
            prev_code_text = json.dumps(compressed, indent=2)
        else:
            prev_code_text = "None yet — this is the first module."

        # Build error section if feedback exists
        error_section = ""
        if error_feedback:
            error_section = CODEGEN_ERROR_SECTION.format(error_feedback=error_feedback)

        user_prompt = CODEGEN_USER.format(
            module_name=module_name,
            architecture_json=json.dumps(architecture, indent=2),
            previous_code=prev_code_text,
            error_section=error_section,
        )

        # Optimize context
        sys_prompt, user_prompt = self.optimizer.optimize_context(
            system_prompt=self.system_prompt(),
            user_prompt=user_prompt,
            context_parts={
                "architecture": json.dumps(architecture, indent=2),
                "previous_code": prev_code_text,
            },
            role="code_generator",
        )

        result = await self._llm_generate_json(user_prompt, temperature=0.2, max_tokens=8192)

        try:
            output = CodeGenOutput(**result)
        except Exception:
            result.setdefault("module_name", module_name)
            result.setdefault("files", [])
            result.setdefault("dependencies", [])
            result.setdefault("notes", "")
            output = CodeGenOutput(**result)

        self._record(message)
        return message.reply(self.role, output.model_dump())
