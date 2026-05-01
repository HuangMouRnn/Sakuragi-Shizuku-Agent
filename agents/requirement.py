"""Requirement Agent: parses natural language into structured specifications."""

from __future__ import annotations

import json

from core.base_agent import BaseAgent
from core.models import AgentMessage, AgentRole, MessageType, RequirementOutput
from prompts.templates import (
    REQUIREMENT_EXAMPLE_INPUT,
    REQUIREMENT_EXAMPLE_OUTPUT,
    REQUIREMENT_SYSTEM,
    REQUIREMENT_USER,
)


class RequirementAgent(BaseAgent):
    role = AgentRole.REQUIREMENT
    name = "RequirementAgent"

    def system_prompt(self) -> str:
        return REQUIREMENT_SYSTEM

    async def execute(self, message: AgentMessage) -> AgentMessage:
        payload = message.payload
        raw_req = payload.get("raw_requirement", "")
        project_name = payload.get("project_name", "untitled")
        constraints = payload.get("constraints", [])

        constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"

        # Build few-shot prompt
        few_shot = f"""## Example

Input: {REQUIREMENT_EXAMPLE_INPUT}

Expected Output:
{REQUIREMENT_EXAMPLE_OUTPUT}

---

Now analyze the actual requirement below."""

        user_prompt = REQUIREMENT_USER.format(
            raw_requirement=raw_req,
            project_name=project_name,
            constraints=constraints_text,
        )

        full_prompt = f"{few_shot}\n\n{user_prompt}"

        result = await self._llm_generate_json(full_prompt, temperature=0.2)

        # Validate and normalize
        try:
            output = RequirementOutput(**result)
        except Exception:
            # Try to fix common issues
            result.setdefault("project_name", project_name)
            result.setdefault("summary", "")
            result.setdefault("user_stories", [])
            result.setdefault("functional_requirements", [])
            result.setdefault("non_functional_requirements", [])
            result.setdefault("tech_constraints", [])
            result.setdefault("assumptions", [])
            output = RequirementOutput(**result)

        self._record(message)
        return message.reply(self.role, output.model_dump())
