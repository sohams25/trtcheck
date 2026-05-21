# CLAUDE.md — trtcheck Workspace

> **Project:** ONNX → TensorRT Pre-Flight Checker  
> **Domain:** Static analysis / developer tooling for edge AI deployment  
> **Language:** Python 3.10+  
> **Key External Dependency:** `onnx` (Open Neural Network Exchange library)

---

## 1. Overview & Engineering Philosophy

**trtcheck** is a static analysis CLI tool that inspects ONNX files and predicts whether they will successfully convert to TensorRT engines — before the developer ever touches a GPU. It diagnoses unsupported operators, precision mismatches, dynamic shape issues, and control flow problems, then suggests specific remediation steps.

**The product thesis:** Every edge AI developer loses 2–6 hours per TensorRT conversion failure in trial-and-error debugging. trtcheck collapses that to 10 seconds with a structured diagnostic report.

**Engineering principles:**
- **Static analysis must be fast.** <10 seconds for a 50MB ONNX file. No excuses.
- **Diagnostics must be actionable.** Every issue includes a specific fix, not just "this failed."
- **Output must be beautiful.** Recruiters and developers judge tools by their CLI output.
- **Zero runtime dependencies on TensorRT.** The tool must work on any laptop without NVIDIA drivers.

---

## 2. Workspace Structure

```
trtcheck/
├── trtcheck/                    # Source package
│   ├── __init__.py              # Version, exports
│   ├── cli.py                   # Click CLI (entry point)
│   ├── analyzer.py              # Core: ONNX → AnalysisReport
│   ├── checkers/                # Modular analysis plugins
│   │   ├── __init__.py
│   │   ├── operator_support.py  # ONNX op vs. TRT support matrix
│   │   ├── precision.py         # INT64, UINT8, BF16 issues
│   │   ├── dynamic_shapes.py    # Dynamic batch/height/width
│   │   ├── control_flow.py      # If/Loop/Scan compatibility
│   │   └── graph_structure.py   # Outputs, isolates, empties
│   ├── reporters/               # Output formatters
│   │   ├── __init__.py
│   │   ├── console.py           # Rich terminal tables
│   │   ├── html.py              # Self-contained HTML report
│   │   └── json.py              # Machine-readable JSON
│   └── data/                    # Bundled static data
│       ├── operator_matrix.json # TRT 8.x–10.x operator support
│       └── remediation_db.json  # Known issue → fix mapping
├── tests/
│   ├── test_analyzer.py
│   ├── test_checkers.py
│   └── fixtures/                # Intentionally broken ONNX files
│       ├── resnet50_clean.onnx
│       ├── failing/
│       │   ├── sequence_empty.onnx
│       │   ├── int64_weights.onnx
│       │   ├── fully_dynamic.onnx
│       │   ├── uint8_input.onnx
│       │   └── control_flow_loop.onnx
│       └── generate_broken.py   # Script to regenerate fixtures
├── setup.py
├── pyproject.toml
├── .github/workflows/ci.yml
└── README.md
```

**Rule:** `checkers/` and `reporters/` are strictly separate. A checker must never import a reporter. The CLI is the only layer that connects them.

---

## 3. Everything-Claude-Code (ECC) Orchestration

### 3.1 Active Subagent Stack

Invoke these subagents by name when delegating tasks.

| Subagent | Role | When to Invoke |
|----------|------|----------------|
| **TDD-Agent** | Writes test specs before implementation | Every new checker, every new reporter, every utility module |
| **Matrix-Keeper** | Maintains `operator_matrix.json` and `remediation_db.json` | When a new TRT version drops; when a new ONNX op is discovered |
| **Reporter-Designer** | Designs CLI/ HTML / JSON output | When adding a new output format or improving existing one |
| **Fixture-Forge** | Creates intentionally broken ONNX test fixtures | When a new failure mode is identified |

### 3.2 Pre-Execution Hook (PreToolUse)

Before generating any code, evaluate:

1. **Is this a new checker or a modification to an existing one?** If new → invoke TDD-Agent first.
2. **Does this respect the checker → reporter → CLI separation?** Checkers must return `List[Issue]`, never print or format.
3. **Will this require updating `operator_matrix.json` or `remediation_db.json`?** If yes, invoke Matrix-Keeper.
4. **Are we handling edge cases in ONNX parsing?** Empty graphs, zero-node graphs, graphs with no outputs.

### 3.3 Post-Execution Hook

After any file modification:

```bash
black <file> && isort <file>
python -m py_compile <file>
```

After checker implementation:
```bash
pytest tests/test_checkers.py -k <checker_name> -v
```

After reporter implementation:
```bash
pytest tests/test_reporters.py -v
```

After data file (JSON) modification:
```bash
python -c "import json; json.load(open('trtcheck/data/operator_matrix.json'))"
pytest tests/test_analyzer.py -v
```

### 3.4 State Store Policy

Append non-obvious ONNX / TensorRT behavior to `.claude_state.jsonl`:

```json
{"timestamp": "2026-05-21T10:00:00Z", "project": "trtcheck", "category": "operator", "lesson": "TRT 10.3 supports Loop only with static trip count. Dynamic trip count fails silently with misleading error.", "affects": "checkers/control_flow.py", "source": "NVIDIA Developer Forums #45678"}
```

This file is read at session start to surface accumulated knowledge.

---

## 4. Development Commands

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run Test Suite

```bash
pytest tests/ -v --cov=trtcheck --cov-report=term-missing
```

### Single Test

```bash
pytest tests/test_checkers.py::test_sequence_empty_detected -v
```

### Type Check

```bash
mypy trtcheck/ --strict
```

### Lint & Format

```bash
black . && isort .
```

### Run CLI Locally

```bash
python -m trtcheck tests/fixtures/resnet50_clean.onnx
python -m trtcheck tests/fixtures/failing/sequence_empty.onnx --verbose
```

### Install in Editable Mode (for ECC sessions)

```bash
pip install -e .
```

---

## 5. Architecture & TDD Mandates

### 5.1 Test-Driven Development — Strict

**Every checker, every reporter, every utility must have tests before implementation.**

The TDD cycle:
1. TDD-Agent writes the test with happy path, boundary, and failure cases.
2. Run the test. Confirm it fails for the right reason.
3. Implement the minimal code to pass.
4. Refactor if needed.
5. Commit: `test: add X` → `feat: implement X`.

**Example — Adding a new checker:**

```python
# Step 1: TDD-Agent writes this FIRST (tests/test_checkers.py)
def test_cast_uint8_to_fp32_detected_as_warning():
    """UINT8 inputs are common in image preprocessing but TensorRT prefers FP32/INT8."""
    model = _create_onnx_with_cast(from_type=TensorProto.UINT8, to_type=TensorProto.FLOAT)
    checker = PrecisionChecker()
    issues = checker.check(model)
    
    uint8_issues = [i for i in issues if "UINT8" in i.message]
    assert len(uint8_issues) == 1
    assert uint8_issues[0].severity == Severity.WARNING
    assert "preprocessing" in uint8_issues[0].remediation.lower()
```

```python
# Step 2: Developer implements this SECOND (trtcheck/checkers/precision.py)
class PrecisionChecker:
    def check(self, model: ModelProto) -> List[Issue]:
        issues = []
        for node in model.graph.node:
            if node.op_type == "Cast":
                # ... detect UINT8 casts ...
                issues.append(Issue(...))
        return issues
```

### 5.2 Checker Architecture

All checkers inherit from a base protocol:

```python
from typing import Protocol, List
import onnx

class Checker(Protocol):
    def check(self, model: onnx.ModelProto) -> List[Issue]:
        ...
```

**Why a protocol, not a base class?** Checkers are pure functions. No shared state. A protocol enforces the interface without forcing inheritance.

**The analyzer composes all checkers:**

```python
class Analyzer:
    def __init__(self, config: AnalyzerConfig):
        self.checkers: List[Checker] = [
            OperatorSupportChecker(config.target_trt),
            PrecisionChecker(),
            DynamicShapeChecker(),
            ControlFlowChecker(config.target_trt),
            GraphStructureChecker(),
        ]
    
    def analyze(self, model: ModelProto) -> AnalysisReport:
        all_issues = []
        for checker in self.checkers:
            all_issues.extend(checker.check(model))
        
        return AnalysisReport(
            total_nodes=len(model.graph.node),
            issues=all_issues,
            # ... compute counts, estimated fusions, etc.
        )
```

### 5.3 Data File Management

`operator_matrix.json` and `remediation_db.json` are version-controlled, hand-curated assets. They are NOT generated at runtime.

**Update process when a new TRT version releases:**
1. NVIDIA publishes release notes with operator changes.
2. Matrix-Keeper updates `operator_matrix.json` with the new version's support status.
3. TDD-Agent adds tests for any newly supported / newly unsupported operators.
4. CI validates that all entries in the JSON are loadable and referenced by tests.

### 5.4 Reporter Architecture

Reporters are pure formatters. Input: `AnalysisReport`. Output: formatted string.

```python
class Reporter(Protocol):
    def render(self, report: AnalysisReport) -> str:
        ...
```

**Console reporter** uses `rich` for colored tables and panels.  
**HTML reporter** produces a self-contained file with no external CSS/JS dependencies.  
**JSON reporter** produces machine-readable output for CI pipelines.

---

## 6. Benchmarking & Performance

### 6.1 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Analysis time (ResNet-50 ONNX, ~23MB) | <10 seconds | `time python -m trtcheck model.onnx` |
| Peak memory (same) | <200MB RSS | `/usr/bin/time -v` |
| True positive rate (known failing models) | >95% | Run against 50 intentionally broken fixtures |
| False positive rate (known good models) | <5% | Run against 10 clean fixtures |

### 6.2 Benchmark Script

```python
# tests/benchmark_performance.py
import time
import trtcheck

def benchmark_analysis_speed():
    """Measure analysis time on standard ONNX models."""
    for name, path in TEST_MODELS.items():
        start = time.perf_counter()
        report = trtcheck.analyze(path)
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.3f}s, {len(report.issues)} issues")
        assert elapsed < 10.0, f"{name} took too long: {elapsed:.2f}s"
```

### 6.3 Fixture Generation

`tests/fixtures/generate_broken.py` programmatically creates ONNX files that exhibit specific failure modes:

```python
def create_sequence_empty_model() -> onnx.ModelProto:
    """Create ONNX with SequenceEmpty op (unsupported in TRT)."""
    # SequenceEmpty appears when PyTorch code uses List[Tensor] = []
    ...

def create_int64_weights_model() -> onnx.ModelProto:
    """Create ONNX with INT64 Constant nodes (TRT casts to INT32, may overflow)."""
    ...

def create_fully_dynamic_shapes_model() -> onnx.ModelProto:
    """Create ONNX with all dynamic dimensions."""
    ...
```

**These fixtures are the test suite's foundation.** They must be deterministic and version-controlled.

---

## 7. CLI Design

### Commands

```bash
# Basic analysis
trtcheck model.onnx

# Specify TRT version
trtcheck model.onnx --target-trt 10.3

# Output formats
trtcheck model.onnx --format console          # Default: rich colored output
trtcheck model.onnx --format json --output report.json
trtcheck model.onnx --format html --output report.html

# Filter by severity
trtcheck model.onnx --severity critical       # Only show blockers
trtcheck model.onnx --severity warning        # Critical + warnings

# Verbose: show info-level too (what WILL work)
trtcheck model.onnx --verbose

# Diff two ONNX files (before/after fix)
trtcheck model_v1.onnx model_v2.onnx --diff

# Version info
trtcheck --version
```

### Console Output Example

```
╔══════════════════════════════════════════════════════════════════════╗
║ trtcheck Report                                                       ║
║ model.onnx → TensorRT 10.3                                           ║
║ Status: CONVERSION WILL FAIL (1 critical, 2 warnings)              ║
╚══════════════════════════════════════════════════════════════════════╝

┌──────────┬────────┬─────────────┬────────────────────────┬─────────────────────────────┐
│ Severity │ Node   │ Operator    │ Issue                  │ Fix                         │
├──────────┼────────┼─────────────┼────────────────────────┼─────────────────────────────┤
│ CRITICAL │ n4     │ SequenceEmpty│ Not supported in TRT  │ Replace List[Tensor] with   │
│          │        │             │ 10.3. Caused by PyTorch│ torch.stack() or pre-alloc. │
│          │        │             │ list operations.       │ See: github.com/.../1044    │
├──────────┼────────┼─────────────┼────────────────────────┼─────────────────────────────┤
│ WARNING  │ n12    │ Cast        │ UINT8 → FLOAT cast may │ Ensure preprocessing casts  │
│          │        │             │ cause precision loss.  │ to FP32 before ONNX export. │
├──────────┼────────┼─────────────┼────────────────────────┼─────────────────────────────┤
│ WARNING  │ input  │ Input       │ Fully dynamic shape:   │ Specify dynamic_axes only  │
│          │        │             │ [batch, 3, h, w].      │ for batch dim in export.    │
└──────────┴────────┴─────────────┴────────────────────────┴─────────────────────────────┘

Summary: 1 critical issue. Estimated fix time: 15-30 minutes.
```

---

## 8. ECC Subagent Prompts

### TDD-Agent

```
You are TDD-Agent for trtcheck. Write pytest test files before any implementation exists.

For the checker described below, produce a complete test file with:
1. Happy path: typical case that should pass or be detected correctly
2. Boundary: empty graph, single-node graph, maximum ONNX opset version
3. Failure mode: the exact failure pattern this checker is designed to catch
4. False positive avoidance: a similar but valid pattern that should NOT trigger

Use small, deterministic ONNX fixtures (create with onnx.helper in the test file).
Do not download external models.
Output: tests/test_{checker_name}.py
```

### Matrix-Keeper

```
You are Matrix-Keeper. Update operator_matrix.json and remediation_db.json.

NVIDIA has released TensorRT {VERSION}. Review their operator support changes:
- Newly supported operators
- Previously partial, now full support
- New limitations or caveats

Update the JSON files accordingly. Add remediation entries for any newly unsupported patterns.
Validate: all JSON must load cleanly, all operators referenced in tests must have entries.
```

### Fixture-Forge

```
You are Fixture-Forge. Create intentionally broken ONNX test fixtures.

Generate a Python script (tests/fixtures/generate_{name}.py) that creates an ONNX model
exhibiting this specific TensorRT failure mode: {FAILURE_MODE_DESCRIPTION}.

The fixture must:
- Be <100KB
- Use only onnx.helper (no PyTorch dependency)
- Be deterministic (same output every run)
- Include a docstring explaining the failure mode

Output: tests/fixtures/failing/{name}.onnx + generation script
```

---

## 9. Quick Reference

| Situation | Action |
|-----------|--------|
| Adding a new checker | Invoke TDD-Agent → write test → implement → run `pytest -k <checker>` |
| NVIDIA releases new TRT version | Invoke Matrix-Keeper → update operator_matrix.json → add tests |
| Console output looks wrong | Check `reporters/console.py` — Rich table syntax |
| Test passes locally, fails in CI | Check Python version, `onnx` package version, fixture paths |
| Want to add HTML report | Invoke Reporter-Designer → implement HTMLReporter → add test |
| Stuck on obscure ONNX behavior | Read `.claude_state.jsonl` → add new entry when resolved |
| Ready to release | Run full test suite → check benchmark targets → write README → tag v1.0 |