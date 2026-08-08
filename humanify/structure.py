"""Structural audit (Phase 3.4): document-level layout and section tells.

Inspects document *structure* (paragraph-length consistency, closing
paragraph, section headers) for AI-shaped layout tells (KB 02):

* T5 -- "suspiciously even" paragraph lengths
* T3 -- hourglass closing with a generic-positive closer
* T2 -- boilerplate challenges/future/conclusion section headers
"""

import re
import statistics

# T5 thresholds
_EVEN_PARAGRAPH_CV = 0.15
_MIN_EVEN_PARAGRAPHS = 4

# T3 -- generic-positive closers that end an essay without a real point.
_HOURGLASS_CLOSERS = (
    "in conclusion",
    "overall, this",
    "overall this",
    "in summary",
    "to summarize",
)

# T2 -- boilerplate section-header words.
_SECTION_HEADER_RE = re.compile(
    r"\b(challenges|future|conclusion|future prospects)\b", re.IGNORECASE
)
_HEADER_MAX_LEN = 100

_SEVERITY_WEIGHT = {"low": 0.2, "medium": 0.4, "high": 0.6}

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def _paragraphs(text: str) -> list[str]:
    """Split into non-empty paragraphs by blank lines."""
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def _paragraph_cv(paragraphs: list[str]) -> float | None:
    """Coefficient of variation of paragraph word lengths (population std/mean)."""
    lengths = [len(p.split()) for p in paragraphs]
    if len(lengths) < 2:
        return None
    mean = statistics.mean(lengths)
    if mean <= 0:
        return None
    return statistics.pstdev(lengths) / mean


def _is_header_like(line: str) -> bool:
    """Heading heuristic: short line not ending in sentence punctuation."""
    line = line.strip()
    if not line or len(line) > _HEADER_MAX_LEN:
        return False
    return not line.endswith((".", "!", "?", "..."))


def _hourglass_closure(final_para: str) -> str | None:
    """Return the matched generic-positive closer phrase or ``None``."""
    lower = " " + final_para.lower().strip() + " "
    for closer in _HOURGLASS_CLOSERS:
        if closer in lower:
            return closer
    return None


def audit_structure(text: str) -> dict:
    """Audit *text* for document-level structural tells.

    Returns ``{"findings": [{"name", "severity", "evidence", "kb_ref"}],
    "score": 0.0-1.0}``. A higher ``score`` indicates more AI-shaped layout.
    """
    paragraphs = _paragraphs(text)
    findings: list[dict] = []

    # T5 -- paragraph-length variance (suspiciously even).
    cv = _paragraph_cv(paragraphs)
    if (
        cv is not None
        and len(paragraphs) >= _MIN_EVEN_PARAGRAPHS
        and cv < _EVEN_PARAGRAPH_CV
    ):
        findings.append({
            "name": "suspiciously_even_paragraphs",
            "severity": "medium",
            "evidence": (
                f"paragraph length CV={cv:.3f} across "
                f"{len(paragraphs)} paragraphs"
            ),
            "kb_ref": "T5",
        })

    # T3 -- hourglass closing in the final paragraph.
    if paragraphs:
        closer = _hourglass_closure(paragraphs[-1])
        if closer:
            findings.append({
                "name": "hourglass_closing",
                "severity": "low",
                "evidence": (
                    f"final paragraph contains generic closer {closer!r}"
                ),
                "kb_ref": "T3",
            })

    # T2 -- boilerplate section headers.
    for line in text.splitlines():
        line = line.strip()
        if not line or not _is_header_like(line):
            continue
        if _SECTION_HEADER_RE.search(line):
            findings.append({
                "name": "challenges_future_section",
                "severity": "medium",
                "evidence": f"header line: {line!r}",
                "kb_ref": "T2",
            })

    score = min(
        sum(_SEVERITY_WEIGHT.get(f["severity"], 0.0) for f in findings), 1.0
    )
    return {"findings": findings, "score": round(score, 3)}
