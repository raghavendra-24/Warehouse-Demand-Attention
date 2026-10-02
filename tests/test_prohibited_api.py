"""No hidden attention implementation (§2.1, §7, A-02, FAC-13).

Scans the package and the experiment scripts (not tests or docs, which name
these APIs on purpose).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "wda").rglob("*.py")) + sorted((ROOT / "experiments").rglob("*.py"))

BANNED_EVERYWHERE = [
    "MultiheadAttention",
    "scaled_dot_product_attention",
    "nn.Transformer",
    "import transformers",
    "from transformers",
    "xformers",
    "flash_attn",
]
BANNED_IN_CORE = ["torch.softmax", "F.softmax", "functional.softmax", "nn.Softmax", ".softmax("]


def test_sources_exist():
    assert any(p.name == "attention.py" for p in SOURCES)


def test_no_prohibited_attention_api_anywhere():
    hits = [(p.relative_to(ROOT), b) for p in SOURCES for b in BANNED_EVERYWHERE if b in p.read_text()]
    assert hits == []


def test_softmax_in_the_core_is_hand_written():
    core = (ROOT / "wda" / "attention.py").read_text()
    assert [b for b in BANNED_IN_CORE if b in core] == []
