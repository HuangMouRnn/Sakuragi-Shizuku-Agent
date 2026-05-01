"""Context window optimization: compression, truncation, and chunking."""

from __future__ import annotations

import json
import logging
import tiktoken
from typing import Any

logger = logging.getLogger(__name__)

# Token budgets per agent role (conservative to leave room for responses)
ROLE_BUDGETS = {
    "requirement": 6000,
    "architect": 8000,
    "code_generator": 10000,
    "test": 8000,
    "debug": 10000,
}
DEFAULT_BUDGET = 8000


class ContextOptimizer:
    """Manages context window size for LLM calls."""

    def __init__(self, model: str = "gpt-4o"):
        self._encoder = None
        self._model = model

    def _get_encoder(self):
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def count_tokens(self, text: str) -> int:
        enc = self._get_encoder()
        return len(enc.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        enc = self._get_encoder()
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = tokens[:max_tokens]
        return enc.decode(truncated) + "\n... [truncated]"

    # ── High-level optimization ──

    def optimize_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context_parts: dict[str, str],
        role: str = "code_generator",
        max_output_tokens: int = 4096,
    ) -> tuple[str, str]:
        """
        Optimize the full context for an LLM call.
        Returns (system_prompt, user_prompt) that fit within budget.
        """
        budget = ROLE_BUDGETS.get(role, DEFAULT_BUDGET)
        # Reserve tokens for output
        available = budget - max_output_tokens - self.count_tokens(system_prompt)
        available = max(available, 2000)

        user_tokens = self.count_tokens(user_prompt)
        context_text = "\n\n".join(f"### {k}\n{v}" for k, v in context_parts.items())
        context_tokens = self.count_tokens(context_text)

        total_needed = user_tokens + context_tokens
        if total_needed <= available:
            return system_prompt, f"{context_text}\n\n---\n\n{user_prompt}"

        # Need to compress context
        logger.info("Context optimization needed: %d tokens > %d budget", total_needed, available)
        optimized_parts = self._compress_parts(context_parts, available - user_tokens)
        optimized_context = "\n\n".join(f"### {k}\n{v}" for k, v in optimized_parts.items())
        return system_prompt, f"{optimized_context}\n\n---\n\n{user_prompt}"

    def _compress_parts(self, parts: dict[str, str], token_budget: int) -> dict[str, str]:
        """Compress context parts to fit within token budget."""
        # Sort by priority (keep recent/important parts longer)
        priority_order = ["error_feedback", "previous_code", "architecture", "requirement", "other"]
        sorted_keys = sorted(
            parts.keys(),
            key=lambda k: next(
                (i for i, p in enumerate(priority_order) if p in k.lower()),
                len(priority_order) - 1,
            ),
        )

        per_part_budget = max(500, token_budget // max(len(sorted_keys), 1))
        result = {}
        remaining = token_budget

        for key in sorted_keys:
            text = parts[key]
            tokens = self.count_tokens(text)
            if tokens <= per_part_budget:
                result[key] = text
                remaining -= tokens
            else:
                # Truncate less important parts more aggressively
                allocated = min(per_part_budget, remaining)
                if allocated < 200:
                    result[key] = f"[{key}: omitted due to context limit]"
                else:
                    result[key] = self.truncate_to_tokens(text, allocated)
                    remaining -= allocated

        return result

    # ── Code-specific compression ──

    @staticmethod
    def compress_code(code: str, keep_signatures: bool = True) -> str:
        """Compress code by removing docstrings and comments while keeping structure."""
        lines = code.split("\n")
        result = []
        in_docstring = False
        in_comment_block = False

        for line in lines:
            stripped = line.strip()

            # Handle triple-quote docstrings
            if '"""' in stripped or "'''" in stripped:
                quote_count = stripped.count('"""') + stripped.count("'''")
                if quote_count >= 2:
                    # Single-line docstring, skip
                    continue
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue

            # Skip comment blocks
            if stripped.startswith("#") and not keep_signatures:
                continue

            # Keep function/class signatures even when compressing
            if keep_signatures and (stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("async def ")):
                result.append(line)
                continue

            result.append(line)

        return "\n".join(result)

    @staticmethod
    def summarize_code_structure(code: str) -> str:
        """Extract just the structure (class/function signatures) from code."""
        lines = code.split("\n")
        structure = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("class ") or stripped.startswith("def ") or stripped.startswith("async def "):
                structure.append(stripped.split(":")[0] + ":")
            elif stripped.startswith("@"):
                structure.append(stripped)

        return "\n".join(structure) if structure else "[empty module]"

    # ── Chunking for large codebases ──

    @staticmethod
    def chunk_codebase(files: dict[str, str], chunk_tokens: int = 3000) -> list[dict[str, str]]:
        """Split a codebase into chunks that fit within token limits."""
        chunks = []
        current_chunk: dict[str, str] = {}
        current_size = 0
        optimizer = ContextOptimizer()

        for path, content in files.items():
            tokens = optimizer.count_tokens(content)
            if tokens > chunk_tokens:
                # File is too large, split it
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = {}
                    current_size = 0
                # Split large file into sub-chunks
                lines = content.split("\n")
                sub_chunk: list[str] = []
                sub_size = 0
                for line in lines:
                    line_tokens = optimizer.count_tokens(line)
                    if sub_size + line_tokens > chunk_tokens and sub_chunk:
                        chunks.append({path + f" (part {len(chunks)})": "\n".join(sub_chunk)})
                        sub_chunk = []
                        sub_size = 0
                    sub_chunk.append(line)
                    sub_size += line_tokens
                if sub_chunk:
                    chunks.append({path + f" (part {len(chunks)})": "\n".join(sub_chunk)})
            elif current_size + tokens > chunk_tokens:
                chunks.append(current_chunk)
                current_chunk = {path: content}
                current_size = tokens
            else:
                current_chunk[path] = content
                current_size += tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
