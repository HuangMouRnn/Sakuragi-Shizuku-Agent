# AutoDev Agent — Multi-Agent Auto Development Framework

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Natural Language Input                    │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Requirement Agent                          │
│  NL → User Stories, Functional/Non-Functional Requirements  │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Architect Agent                          │
│  Spec → Tech Stack, Modules, Directory Structure, API Design│
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│               For Each Module (modular generation)           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  CodeGen Agent → Test Agent → Debug Agent             │  │
│  │                    ↑           │                       │  │
│  │                    └───────────┘  (self-healing loop)  │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Integration Test → Output Project               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
RequirementInput → RequirementOutput
                         ↓
                   ArchitectInput → ArchitectOutput
                         ↓
                   CodeGenInput → CodeGenOutput (per module)
                         ↓
                   TestInput → TestOutput
                         ↓ (if failures)
                   DebugInput → DebugOutput → patches applied → retest
```

### Shared Infrastructure

- **MemoryManager**: Cross-agent context sharing, log compression, disk persistence
- **ContextOptimizer**: Token counting, truncation, code compression, chunking
- **Orchestrator**: Agent lifecycle, message routing, pipeline state, event emission

---

## 2. Agent Definitions

### 2.1 Requirement Agent

| Field | Value |
|-------|-------|
| **Input** | `{ raw_requirement: str, project_name: str, constraints: str[] }` |
| **Output** | `{ project_name, summary, user_stories[], functional_requirements[], non_functional_requirements[], tech_constraints[], assumptions[] }` |
| **Prompt Strategy** | Few-shot with example NL→spec conversion |

### 2.2 Architect Agent

| Field | Value |
|-------|-------|
| **Input** | `{ requirement: RequirementOutput, preferred_stack: str[] }` |
| **Output** | `{ project_name, tech_stack, modules[], directory_structure, data_models[], api_endpoints[], sequence_diagram }` |
| **Prompt Strategy** | Few-shot with example architecture design |

### 2.3 Code Generator Agent

| Field | Value |
|-------|-------|
| **Input** | `{ architecture, module_name, previous_code{}, error_feedback? }` |
| **Output** | `{ module_name, files[{path, content, language}], dependencies[], notes }` |
| **Prompt Strategy** | Context-optimized, error-aware regeneration |

### 2.4 Test Agent

| Field | Value |
|-------|-------|
| **Input** | `{ code_files[], requirement, architecture, previous_failures[] }` |
| **Output** | `{ test_cases[], results[], coverage_pct, all_passed, failure_summary }` |
| **Prompt Strategy** | Generates pytest tests, then actually executes them in a temp environment |

### 2.5 Debug Agent

| Field | Value |
|-------|-------|
| **Input** | `{ failed_tests[], source_code{}, test_code{}, architecture, attempt_number, max_attempts }` |
| **Output** | `{ root_cause, patches[{file_path, original_snippet, fixed_snippet, explanation}], reasoning_chain[], confidence, needs_new_tests }` |
| **Prompt Strategy** | Chain-of-Thought reasoning with step-by-step analysis |

---

## 3. Core Execution Flow (Pseudocode)

```python
async def pipeline(requirement_text, project_name):
    # Phase 1: Requirement Analysis
    req = await requirement_agent.analyze(requirement_text, project_name)

    # Phase 2: Architecture Design
    arch = await architect_agent.design(req)

    # Phase 3: Per-module Code Generation + Test + Debug
    all_code = {}
    for module in arch.modules:
        for iteration in range(MAX_DEBUG_ITERATIONS):
            # Generate code
            code = await codegen_agent.generate(
                module=module,
                architecture=arch,
                previous_code=all_code,
                error_feedback=error_feedback,  # None on first attempt
            )
            all_code.update(code)

            # Run tests
            test_result = await test_agent.test(code, req, arch)

            if test_result.all_passed:
                break  # Module done

            # Debug and patch
            debug_result = await debug_agent.analyze(
                failures=test_result.failures,
                source=all_code,
                attempt=iteration + 1,
            )

            # Apply patches to code
            for patch in debug_result.patches:
                all_code[patch.file] = apply_patch(all_code[patch.file], patch)

            error_feedback = debug_result.root_cause

    # Phase 4: Integration Test
    final_test = await test_agent.test(all_code, req, arch)

    # Write output
    write_project(all_code, arch)
    return final_test.all_passed
```

---

## 4. Key Design Decisions

### Modular Code Generation
Instead of generating an entire project at once (which exceeds context limits), the system generates one module at a time, passing previously generated code as context for consistency.

### Self-Healing Loop
The `test → debug → patch → retest` cycle runs up to N times per module. Temperature increases slightly on each retry to explore different fix strategies.

### Context Optimization
- **Token counting** via tiktoken prevents context overflow
- **Code compression** removes docstrings/comments from context while keeping signatures
- **Priority-based truncation** keeps error feedback and recent code longer than historical context
- **Chunking** splits large codebases into manageable pieces

### Memory Management
- **Shared memory** allows agents to read each other's outputs
- **Agent buffers** maintain append-only logs for debugging
- **LRU eviction** prevents unbounded memory growth
- **Disk persistence** survives process restarts

---

## 5. Project Structure

```
mimo100t/
├── agents/
│   ├── __init__.py
│   ├── requirement.py      # Requirement Agent
│   ├── architect.py         # Architect Agent
│   ├── code_generator.py    # Code Generator Agent
│   ├── test_agent.py        # Test Agent (generates + runs tests)
│   └── debug_agent.py       # Debug Agent (Chain-of-Thought analysis)
├── core/
│   ├── __init__.py
│   ├── models.py            # Pydantic data models for all inter-agent messages
│   ├── base_agent.py        # Abstract base class + unified LLM client
│   ├── memory_manager.py    # Shared memory with eviction + persistence
│   ├── context_optimizer.py # Token counting, compression, chunking
│   └── orchestrator.py      # Pipeline orchestrator
├── memory/
│   └── __init__.py
├── prompts/
│   ├── __init__.py
│   └── templates.py         # All system/user prompts + few-shot examples
├── runner/
│   ├── __init__.py
│   ├── debug_loop.py        # Self-healing debug loop implementation
│   └── cli.py               # Typer CLI with Rich output
├── tests/
│   ├── __init__.py
│   ├── test_models.py       # Data model tests
│   ├── test_memory.py       # Memory + context optimizer tests
│   └── test_orchestrator.py # Orchestrator integration tests
├── main.py                   # Entry point
├── pyproject.toml            # Project config + dependencies
└── ARCHITECTURE.md           # This file
```

---

## 6. Token Optimization Strategy

| Technique | Where Used | Effect |
|-----------|-----------|--------|
| **Signature extraction** | Context passing between modules | ~70% reduction |
| **Docstring removal** | Code in prompts | ~30% reduction |
| **Priority truncation** | Context optimizer | Keeps important parts |
| **Chunking** | Large codebases | Fits within limits |
| **Log compression** | Memory manager | Prevents buffer bloat |
| **Per-role budgets** | Context optimizer | Prevents any single agent from hogging context |

---

## 7. Running

```bash
# Install
cd mimo100t
pip install -e .

# Set API key
export ANTHROPIC_API_KEY=sk-...

# Build a project
python main.py build "Build a REST API for a todo list with SQLite storage" --name todo-api

# Show architecture
python main.py explain

# Check output
python main.py status
```
