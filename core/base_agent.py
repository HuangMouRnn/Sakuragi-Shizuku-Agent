"""Base agent class with LLM integration."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel

from .models import AgentMessage, AgentRole, MessageType

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self, provider: str = "anthropic", model: str | None = None, api_key: str | None = None):
        self.provider = provider
        self.model = model or self._default_model()
        self._api_key = api_key
        self._client = None

    def _default_model(self) -> str:
        defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
        }
        return defaults.get(self.provider, "claude-sonnet-4-20250514")

    def _get_client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            elif self.provider == "openai":
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    async def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        client = self._get_client()

        if self.provider == "anthropic":
            resp = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            return resp.content[0].text

        elif self.provider == "openai":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}] + messages,
            }
            if response_format:
                kwargs["response_format"] = response_format
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        raise ValueError(f"Unsupported provider: {self.provider}")


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    role: AgentRole
    name: str

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._history: list[AgentMessage] = []

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    async def execute(self, message: AgentMessage) -> AgentMessage:
        ...

    async def _llm_generate(
        self,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        return await self.llm.chat(
            system=self.system_prompt(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def _llm_generate_json(
        self,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = await self._llm_generate(
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            inside = False
            for line in lines:
                if line.strip().startswith("```") and not inside:
                    inside = True
                    continue
                elif line.strip() == "```" and inside:
                    break
                elif inside:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        # Try to find JSON object or array
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

        return json.loads(text)

    def _record(self, msg: AgentMessage):
        self._history.append(msg)
        if len(self._history) > 50:
            self._history = self._history[-30:]
