"""CLI entry point for the auto-dev agent framework."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from agents.architect import ArchitectAgent
from agents.code_generator import CodeGeneratorAgent
from agents.debug_agent import DebugAgent
from agents.requirement import RequirementAgent
from agents.test_agent import TestAgent
from core.base_agent import LLMClient
from core.memory_manager import MemoryManager
from core.models import AgentRole
from core.orchestrator import Orchestrator

app = typer.Typer(help="AutoDev Agent: Multi-Agent Auto Development Framework")
console = Console()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def build(
    requirement: str = typer.Argument(..., help="Natural language requirement for the project"),
    project_name: str = typer.Option("auto-project", "--name", "-n", help="Project name"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider (anthropic/openai)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name"),
    max_iterations: int = typer.Option(3, "--max-iter", "-i", help="Max debug iterations per module"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Build a project from a natural language requirement."""
    setup_logging(verbose)

    console.print(Panel.fit(
        "[bold cyan]AutoDev Agent[/bold cyan]\n"
        f"Requirement: {requirement[:80]}...\n"
        f"Provider: {provider} | Max iterations: {max_iterations}",
        title="Starting Build",
    ))

    asyncio.run(_run_build(requirement, project_name, output_dir, provider, model, max_iterations))


async def _run_build(
    requirement: str,
    project_name: str,
    output_dir: str,
    provider: str,
    model: str | None,
    max_iterations: int,
):
    # Initialize components
    llm = LLMClient(provider=provider, model=model)
    memory = MemoryManager()

    # Register all agents
    orchestrator = Orchestrator(
        llm=llm,
        memory=memory,
        output_dir=output_dir,
        max_debug_iterations=max_iterations,
    )

    orchestrator.register(RequirementAgent(llm))
    orchestrator.register(ArchitectAgent(llm))
    orchestrator.register(CodeGeneratorAgent(llm))
    orchestrator.register(TestAgent(llm))
    orchestrator.register(DebugAgent(llm))

    # Rich progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        current_task = None

        async def on_event(event: str, data: dict):
            nonlocal current_task

            if event == "phase_start":
                phase = data.get("phase", "")
                labels = {
                    "requirement": "Analyzing requirements...",
                    "architect": "Designing architecture...",
                    "integration_test": "Running integration tests...",
                }
                if current_task:
                    progress.update(current_task, completed=True)
                current_task = progress.add_task(
                    labels.get(phase, f"Phase: {phase}..."),
                    total=None,
                )

            elif event == "module_start":
                module = data.get("module", "")
                idx = data.get("index", 0) + 1
                total = data.get("total", 1)
                if current_task:
                    progress.update(current_task, completed=True)
                current_task = progress.add_task(
                    f"Module [{idx}/{total}]: {module} — generating code...",
                    total=None,
                )

            elif event == "test_complete":
                passed = data.get("passed", False)
                status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
                console.print(f"  Tests: {status}")

            elif event == "debug_applied":
                cause = data.get("root_cause", "")[:80]
                patches = data.get("patches", 0)
                console.print(f"  Debug: {cause} ({patches} patches)")

            elif event == "pipeline_complete":
                if current_task:
                    progress.update(current_task, completed=True)

        orchestrator.on(on_event)
        state = await orchestrator.run(requirement, project_name)

    # Display results
    _display_results(state, output_dir, project_name)


def _display_results(state, output_dir: str, project_name: str):
    if state.status == "success":
        console.print(Panel.fit(
            "[bold green]BUILD SUCCESSFUL[/bold green]",
            title="Result",
        ))
    else:
        console.print(Panel.fit(
            "[bold red]BUILD FAILED[/bold red]",
            title="Result",
        ))

    # Show generated structure
    project_dir = Path(output_dir) / project_name
    if project_dir.exists():
        tree = Tree(f"[bold]{project_name}/[/bold]")
        _build_tree(tree, project_dir)
        console.print(tree)

    # Show stats
    table = Table(title="Pipeline Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Status", state.status)
    table.add_row("Modules", str(state.total_modules))
    table.add_row("Debug Iterations", str(state.iteration))
    table.add_row("Debug History", str(len(state.debug_history)))
    table.add_row("Files Generated", str(len(state.generated_code)))
    console.print(table)


def _build_tree(tree: Tree, path: Path, depth: int = 0):
    if depth > 4:
        return
    try:
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return
    for item in items:
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            branch = tree.add(f"[bold blue]{item.name}/[/bold blue]")
            _build_tree(branch, item, depth + 1)
        else:
            size = item.stat().st_size
            tree.add(f"{item.name} ({size}B)")


@app.command()
def explain():
    """Show the system architecture and agent descriptions."""
    console.print(Panel.fit(
        "[bold]AutoDev Agent Architecture[/bold]\n\n"
        "┌─────────────────────────────────────────────┐\n"
        "│           Natural Language Input             │\n"
        "└──────────────────┬──────────────────────────┘\n"
        "                   ▼\n"
        "┌──────────────────────────────────────────────┐\n"
        "│  Requirement Agent                           │\n"
        "│  NL → Structured Spec (User Stories, FRs)    │\n"
        "└──────────────────┬───────────────────────────┘\n"
        "                   ▼\n"
        "┌──────────────────────────────────────────────┐\n"
        "│  Architect Agent                             │\n"
        "│  Spec → Modules, API, Directory Structure    │\n"
        "└──────────────────┬───────────────────────────┘\n"
        "                   ▼\n"
        "┌──────────────────────────────────────────────┐\n"
        "│  For each module:                            │\n"
        "│  ┌─────────────────────────────────────────┐ │\n"
        "│  │ CodeGen → Test → [Debug → CodeGen]*     │ │\n"
        "│  │         (self-healing loop)              │ │\n"
        "│  └─────────────────────────────────────────┘ │\n"
        "└──────────────────┬───────────────────────────┘\n"
        "                   ▼\n"
        "┌──────────────────────────────────────────────┐\n"
        "│  Integration Test → Output Project           │\n"
        "└──────────────────────────────────────────────┘",
        title="System Architecture",
    ))


@app.command()
def status(
    output_dir: str = typer.Option("./output", "--output", "-o"),
):
    """Show status of generated projects."""
    out = Path(output_dir)
    if not out.exists():
        console.print("[yellow]No output directory found.[/yellow]")
        return

    for project in sorted(out.iterdir()):
        if project.is_dir():
            files = list(project.rglob("*.py"))
            console.print(f"[bold]{project.name}[/bold]: {len(files)} Python files")


if __name__ == "__main__":
    app()
