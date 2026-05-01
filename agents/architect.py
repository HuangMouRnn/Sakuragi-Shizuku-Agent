"""Architect Agent: designs modular architecture from requirements."""

from __future__ import annotations

import json

from core.base_agent import BaseAgent
from core.models import AgentMessage, AgentRole, ArchitectOutput
from prompts.templates import (
    ARCHITECT_EXAMPLE_INPUT,
    ARCHITECT_EXAMPLE_OUTPUT,
    ARCHITECT_SYSTEM,
    ARCHITECT_USER,
)


class ArchitectAgent(BaseAgent):
    role = AgentRole.ARCHITECT
    name = "ArchitectAgent"

    def system_prompt(self) -> str:
        return ARCHITECT_SYSTEM

    async def execute(self, message: AgentMessage) -> AgentMessage:
        payload = message.payload
        requirement = payload.get("requirement", {})
        preferred_stack = payload.get("preferred_stack", [])

        stack_text = ", ".join(preferred_stack) if preferred_stack else "Choose the best fit"

        few_shot = f"""## Example

Input: {ARCHITECT_EXAMPLE_INPUT}

Expected Output:
{ARCHITECT_EXAMPLE_OUTPUT}

---

Now design the architecture for the actual requirement below."""

        user_prompt = ARCHITECT_USER.format(
            requirement_json=json.dumps(requirement, indent=2),
            preferred_stack=stack_text,
        )

        full_prompt = f"{few_shot}\n\n{user_prompt}"
        result = await self._llm_generate_json(full_prompt, temperature=0.2)

        try:
            output = ArchitectOutput(**result)
        except Exception:
            result.setdefault("project_name", requirement.get("project_name", "project"))
            result.setdefault("tech_stack", {})
            result.setdefault("modules", [])
            result.setdefault("directory_structure", {})
            output = ArchitectOutput(**result)

        self._record(message)
        return message.reply(self.role, output.model_dump())
