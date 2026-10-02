"""No hidden attention implementation (§2.1, §7, A-02, FAC-13).

Parses the package and the experiment scripts (not tests or docs, which name
these APIs on purpose) and inspects every import and attribute access, so
aliases such as `from torch.nn.functional import softmax as s` are caught.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "wda").rglob("*.py")) + sorted((ROOT / "experiments").rglob("*.py"))

BANNED_NAMES = {"MultiheadAttention", "scaled_dot_product_attention", "multi_head_attention_forward",
                "flex_attention", "Transformer", "TransformerEncoder", "TransformerEncoderLayer",
                "TransformerDecoder", "TransformerDecoderLayer"}
BANNED_MODULES = ("transformers", "xformers", "flash_attn", "torch.nn.attention")
BANNED_IN_CORE = {"softmax", "log_softmax", "Softmax", "LogSoftmax", "softmin"}


def names_used(path: Path) -> tuple:
    """(imported module names, imported or accessed attribute/function names)."""
    tree = ast.parse(path.read_text())
    modules, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return modules, names


def test_sources_exist():
    assert any(p.name == "attention.py" for p in SOURCES)


def test_no_prohibited_attention_api_anywhere():
    hits = []
    for p in SOURCES:
        modules, names = names_used(p)
        hits += [(str(p.relative_to(ROOT)), n) for n in names & BANNED_NAMES]
        hits += [(str(p.relative_to(ROOT)), m) for m in modules if m.startswith(BANNED_MODULES)]
    assert hits == []


def test_softmax_in_the_core_is_hand_written():
    _, names = names_used(ROOT / "wda" / "attention.py")
    assert names & BANNED_IN_CORE == set()
    assert "stable_softmax" in names
