# PROJECT 1: trtcheck — ONNX → TensorRT Pre-Flight Checker

## 1. Problem Statement (The Pain You Are Solving)

Every edge AI developer follows this workflow:

```
PyTorch model → ONNX export → TensorRT conversion → .engine file
                              ↑ THIS STEP FAILS 80% OF THE TIME
```

When TensorRT conversion fails, the error messages are cryptic. Examples from real GitHub issues:

| Error Message | What It Actually Means | Root Cause |
|--------------|----------------------|------------|
| `UNSUPPORTED_NODE: SequenceEmpty` | Your model uses a PyTorch list `[]` which ONNX represents as `SequenceEmpty` | PyTorch `List[Tensor] = []` in forward() |
| `Assertion failed: convert_dtype: UINT8` | ONNX has UINT8 tensors but TensorRT only supports FP32, FP16, INT32, INT8 | Image preprocessing with `np.uint8` |
| `at least 5 dimensions are required for input` | MaxPool receives a tensor with fewer than 5 dims after shape inference | Dynamic batch size not handled |
| `INT64 weights detected while TensorRT does not natively support INT64` | PyTorch defaults to INT64 for indices, ONNX preserves it | `torch.LongTensor` for argmax/indices |
| `Network must have at least one output` | Graph optimization removed all outputs or shape inference failed | Control flow (If/Loop) with dynamic shapes |

**The developer workflow today:**
1. Run `trtexec --onnx=model.onnx`
2. Wait 2-5 minutes for it to fail
3. Parse a cryptic C++ error message
4. Google the error → find a forum post from 2019
5. Try a fix → repeat from step 1
6. Average time to resolve: **2-6 hours** per failure

**trtcheck replaces this with:**
1. Run `trtcheck model.onnx`
2. Get a structured report in 10 seconds
3. Follow specific remediation steps
4. Fix the ONNX file, THEN attempt TensorRT conversion

## 2. Technical Architecture

```
trtcheck/
├── trtcheck/
│   ├── __init__.py              # Version, exports
│   ├── cli.py                   # Click-based CLI (main entry point)
│   ├── analyzer.py              # Core: ONNX file → AnalysisReport
│   ├── checkers/                # Modular check plugins
│   │   ├── __init__.py
│   │   ├── operator_support.py  # Check each op against TRT support matrix
│   │   ├── precision.py         # Check INT64, UINT8, BF16 issues
│   │   ├── dynamic_shapes.py    # Check dynamic batch/height/width
│   │   ├── control_flow.py      # Check If/Loop/Scan compatibility
│   │   ├── graph_structure.py   # Check outputs, isolates, empty graphs
│   │   └── fusion_opportunity.py# Identify what TRT will fuse (info only)
│   ├── reporters/               # Output formatters
│   │   ├── __init__.py
│   │   ├── console.py           # Rich terminal output with colors
│   │   ├── html.py              # Self-contained HTML report
│   │   └── json.py              # Machine-readable JSON output
│   └── data/                    # Bundled data files
│       ├── operator_matrix.json # TRT 8.x-10.x operator support matrix
│       └── remediation_db.json  # Known issue → fix mapping
├── tests/
│   ├── test_analyzer.py
│   ├── test_checkers.py
│   └── fixtures/                # Sample ONNX files (small, for testing)
│       ├── resnet50_minimal.onnx
│       ├── yolov8n_minimal.onnx
│       └── failing_models/      # Intentionally broken ONNX files
├── .github/workflows/ci.yml
├── setup.py
├── pyproject.toml
└── README.md
```

## 3. Core Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum

class Severity(str, Enum):
    CRITICAL = "critical"   # Conversion will definitely fail
    WARNING = "warning"     # May fail or cause accuracy issues
    INFO = "info"           # FYI — no action needed

class CheckCategory(str, Enum):
    OPERATOR_SUPPORT = "operator_support"
    PRECISION = "precision"
    DYNAMIC_SHAPES = "dynamic_shapes"
    CONTROL_FLOW = "control_flow"
    GRAPH_STRUCTURE = "graph_structure"

@dataclass
class Issue:
    """A single detected issue."""
    severity: Severity
    category: CheckCategory
    node_name: str              # Which ONNX node (e.g., "n4", "conv_1")
    operator: str               # ONNX op type (e.g., "SequenceEmpty", "Cast")
    message: str                # Human-readable description
    remediation: str            # Specific fix suggestion
    docs_link: Optional[str]    # Link to TRT docs or GitHub issue

@dataclass
class AnalysisReport:
    """Complete analysis of an ONNX file."""
    filename: str
    onnx_ir_version: str
    opset_version: int
    producer: str               # "pytorch", "tensorflow", etc.
    
    # Summary counts
    total_nodes: int
    issues: List[Issue] = field(default_factory=list)
    
    # Categorized
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    # What will TensorRT likely do
    estimated_fusions: List[str] = field(default_factory=list)
    estimated_precision: Dict[str, int] = field(default_factory=dict)  # "FP32": 50, "INT8": 10
    
    # Verdict
    conversion_likely: bool = False  # Overall: will it convert?
    estimated_fix_time: str = ""     # "< 15 minutes", "1-2 hours", etc.
    
    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        ...
```

## 4. The Compatibility Matrix (operator_matrix.json)

This is the crown jewel. A structured JSON file mapping ONNX operators to TensorRT version support:

```json
{
  "schema_version": "1.0",
  "last_updated": "2026-05-21",
  "operators": {
    "Conv": {
      "support": {
        "8.0": "supported",
        "8.6": "supported",
        "10.0": "supported",
        "10.3": "supported"
      },
      "notes": "Full support. NCHW format.",
      "limitations": ["3D Conv requires TRT 8.6+"]
    },
    "SequenceEmpty": {
      "support": {
        "8.0": "not_supported",
        "8.6": "not_supported",
        "10.0": "not_supported",
        "10.3": "not_supported"
      },
      "notes": "Sequence ops not supported. Caused by PyTorch List[Tensor].",
      "remediation": "Replace List[Tensor] with torch.stack() or pre-allocate tensor.",
      "github_issue": "https://github.com/onnx/onnx-tensorrt/issues/1044"
    },
    "Loop": {
      "support": {
        "8.0": "partial",
        "8.6": "partial",
        "10.0": "supported",
        "10.3": "supported"
      },
      "notes": "Supported in TRT 10+. Requires static trip count.",
      "limitations": ["Dynamic trip count not supported", "Nested loops not supported"]
    },
    "Cast": {
      "support": {
        "8.0": "partial",
        "8.6": "partial",
        "10.0": "supported",
        "10.3": "supported"
      },
      "notes": "UINT8 → INT8/INT32 cast may fail in older TRT.",
      "remediation": "Ensure source dtype is FP32/FP16/INT32/INT8 before export."
    },
    "GroupNormalization": {
      "support": {
        "8.0": "not_supported",
        "8.6": "not_supported",
        "10.0": "supported",
        "10.3": "supported"
      },
      "notes": "Added in TRT 10.0. Use BatchNorm for older TRT.",
      "remediation": "Replace nn.GroupNorm with nn.BatchNorm or upgrade TRT to 10.0+."
    }
  }
}
```

**How to build this:** Parse the official [ONNX-TensorRT operator support documentation](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md) programmatically. Update quarterly as new TRT releases come out.

## 5. Each Checker Plugin (Detailed Spec)

### 5.1 OperatorSupportChecker

**Input:** ONNX GraphProto  
**Output:** List[Issue] for each unsupported operator

```python
class OperatorSupportChecker:
    def __init__(self, matrix_path: str, target_trt_version: str = "10.3"):
        self.matrix = json.load(open(matrix_path))
        self.target = target_trt_version
    
    def check(self, model: onnx.ModelProto) -> List[Issue]:
        issues = []
        for node in model.graph.node:
            op = node.op_type
            support = self.matrix["operators"].get(op, {}).get("support", {})
            status = support.get(self.target, "unknown")
            
            if status == "not_supported":
                issues.append(Issue(
                    severity=Severity.CRITICAL,
                    category=CheckCategory.OPERATOR_SUPPORT,
                    node_name=node.name,
                    operator=op,
                    message=f"Operator '{op}' is not supported in TensorRT {self.target}.",
                    remediation=self.matrix["operators"][op].get("remediation", "Search for workaround."),
                    docs_link=self.matrix["operators"][op].get("github_issue")
                ))
            elif status == "partial":
                issues.append(Issue(... severity=Severity.WARNING ...))
        return issues
```

### 5.2 PrecisionChecker

Detects precision issues before TRT sees them:

| Check | Description | Remediation |
|-------|-------------|-------------|
| INT64 weights | PyTorch defaults to int64 for indices | Cast to int32 before ONNX export |
| UINT8 inputs | Image data as uint8 | Cast to float32 in preprocessing |
| BF16 unsupported | bf16 in ONNX but TRT 8.x doesn't support | Use FP16 instead for old TRT |
| Double (FLOAT64) | Rare but happens | Cast to FP32 |
| String tensors | String processing models | Not supported in TRT — use integer encoding |

### 5.3 DynamicShapeChecker

```python
class DynamicShapeChecker:
    def check(self, model: onnx.ModelProto) -> List[Issue]:
        """Check for dynamic shapes that TRT handles poorly."""
        issues = []
        
        for input_tensor in model.graph.input:
            shape = [d.dim_param if d.dim_param else d.dim_value 
                     for d in input_tensor.type.tensor_type.shape.dim]
            
            # Check for fully dynamic shapes
            if all(isinstance(d, str) for d in shape):
                issues.append(Issue(
                    severity=Severity.WARNING,
                    category=CheckCategory.DYNAMIC_SHAPES,
                    node_name=input_tensor.name,
                    operator="Input",
                    message=f"Input '{input_tensor.name}' has fully dynamic shape: {shape}.",
                    remediation="Use torch.onnx.export with example_inputs and dynamic_axes specified only for batch dimension.",
                    docs_link="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes"
                ))
        return issues
```

### 5.4 ControlFlowChecker

Checks If/Loop/Scan nodes for known TRT limitations:
- Loop: trip count must be static or inferable
- If: both branches must produce identical output shapes
- Scan: sequence length must be static

### 5.5 GraphStructureChecker

Basic sanity checks:
- Graph has at least one output
- No isolated nodes (nodes with no connections)
- No duplicate node names
- Constant folding opportunities (large Constant nodes)

## 6. CLI Interface

```bash
# Basic check
trtcheck model.onnx

# Specify target TensorRT version
trtcheck model.onnx --target-trt 10.3

# Output formats
trtcheck model.onnx --format console          # Default: rich terminal
trtcheck model.onnx --format json --output report.json
trtcheck model.onnx --format html --output report.html

# Show only critical issues
trtcheck model.onnx --severity critical

# Verbose: show what WILL work (info level)
trtcheck model.onnx --verbose

# Compare two ONNX files (before/after fix)
trtcheck model_before.onnx model_after.onnx --diff

# Auto-fix mode (apply simple fixes)
trtcheck model.onnx --fix --output model_fixed.onnx
```

## 7. Console Output Example (The "Wow" Factor)

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def render_report(report: AnalysisReport):
    # Header panel with verdict
    if report.conversion_likely:
        header = Panel(
            f"[green bold]✓ LIKELY TO CONVERT[/green bold]\n"
            f"{report.critical_count} critical, {report.warning_count} warnings, {report.info_count} info",
            title="trtcheck Report",
            border_style="green"
        )
    else:
        header = Panel(
            f"[red bold]✗ CONVERSION WILL FAIL[/red bold]\n"
            f"{report.critical_count} critical issues found. Estimated fix time: {report.estimated_fix_time}",
            title="trtcheck Report",
            border_style="red"
        )
    console.print(header)
    
    # Issues table
    table = Table(title="Detected Issues")
    table.add_column("Severity", style="bold")
    table.add_column("Node")
    table.add_column("Operator")
    table.add_column("Issue")
    table.add_column("Fix")
    
    for issue in report.issues:
        sev_color = {"critical": "red", "warning": "yellow", "info": "blue"}[issue.severity.value]
        table.add_row(
            f"[{sev_color}]{issue.severity.value.upper()}[/{sev_color}]",
            issue.node_name,
            issue.operator,
            issue.message,
            issue.remediation
        )
    console.print(table)
    
    # Summary stats
    console.print(f"\n[bold]Total nodes:[/bold] {report.total_nodes}")
    console.print(f"[bold]Estimated fusions:[/bold] {', '.join(report.estimated_fusions)}")
```

## 8. Build Plan (4 Weeks with Claude)

| Week | Component | Claude Prompt |
|------|-----------|--------------|
| **1** | **Data layer** | "Build the operator_matrix.json by parsing the official ONNX-TensorRT operator support docs. Include 50+ operators with support status for TRT 8.0, 8.6, 10.0, 10.3. Also build remediation_db.json with 20+ known issues mapped to fixes." |
| **2** | **Core analyzer + checkers** | "Write the analyzer.py that loads an ONNX model and runs 5 checker plugins: operator support, precision, dynamic shapes, control flow, graph structure. Each checker returns a list of Issue dataclasses. Combine into an AnalysisReport." |
| **3** | **Reporters + CLI** | "Write console reporter using `rich` library with colored tables and panels. Write HTML reporter that produces a self-contained HTML file. Write Click CLI with all flags." |
| **4** | **Tests + docs + release** | "Write 10 unit tests with intentionally broken ONNX fixtures. Write README with installation, usage, examples. Set up GitHub Actions CI. Publish to PyPI." |

## 9. Testing Strategy

Create **intentionally broken ONNX files** as test fixtures:

```python
# test/fixtures/generate_broken_models.py
"""Generate ONNX files that exhibit specific TensorRT failure modes."""

def create_sequence_empty_model() -> onnx.ModelProto:
    """Model with SequenceEmpty op (TRT fails)."""
    # Creates: List[Tensor] = [] → SequenceEmpty → SequenceInsert
    ...

def create_int64_weights_model() -> onnx.ModelProto:
    """Model with INT64 weights (TRT casts, may overflow)."""
    # Creates: Constant with int64 data type
    ...

def create_fully_dynamic_shapes_model() -> onnx.ModelProto:
    """Model with fully dynamic input shapes."""
    # Creates: input with shape ["batch", "channels", "height", "width"]
    ...

def create_uint8_input_model() -> onnx.ModelProto:
    """Model with UINT8 input tensor."""
    # Creates: input with type tensor(uint8)
    ...
```

**Test cases:**
- `test_resnet50_clean` → 0 issues (baseline)
- `test_sequence_empty` → 1 critical (SequenceEmpty)
- `test_int64_weights` → 1 warning (INT64)
- `test_dynamic_shapes` → 1 warning (fully dynamic)
- `test_uint8_input` → 1 critical (UINT8)
- `test_control_flow_loop` → 1 warning (Loop with dynamic trip)
- `test_combined_issues` → multiple issues of different severities
