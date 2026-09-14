"""Split page text into section-aware chunks on clean boundaries.

Rules (validated across news / how-to / docs / code categories):
- Split only on paragraph boundaries (\n\n); never inside a paragraph.
- Track the most recent markdown heading; each chunk carries its heading.
- A single long section packs its paragraphs up to the target (paragraph
  is the hard floor); pathological no-\n\n text falls back to sentence
  ends, then to a hard cut.
- Texts already under the target are returned as a single part.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _heading_label(paragraph: str) -> str:
    """Heading text if the paragraph starts with a markdown heading, else ''."""
    stripped = paragraph.strip()
    if not stripped:
        return ""
    first = stripped.splitlines()[0]
    if _HEADING_RE.match(first):
        return _HEADING_RE.sub("", first).strip().rstrip("¶").strip()
    return ""


def _split_sentences(paragraph: str, target: int) -> list[str]:
    """Last-resort split of a single oversized paragraph."""
    sentences = _SENTENCE_END.split(paragraph)
    if len(sentences) <= 1:
        # no sentence boundaries either: hard cut
        return [paragraph[i : i + target] for i in range(0, len(paragraph), target)]
    out: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > target:
            out.append(cur.strip())
            cur = ""
        cur = (cur + " " + s).strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def _pack(
    paragraphs: list[tuple[str, str]], target: int, head0: str
) -> list[tuple[str, str]]:
    """Pack paragraphs into chunks <= target; never split a paragraph.

    Each paragraph is pre-labeled with the heading it belongs to, so a
    flush triggered by the next section's first paragraph still carries
    the heading of the content it contains.
    """
    chunks: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_len = 0
    cur_head = head0
    for head, p in paragraphs:
        if cur and cur_len + len(p) + 2 > target:
            chunks.append((cur_head, "\n\n".join(cur)))
            cur, cur_len = [], 0
            cur_head = head
        elif not cur:
            cur_head = head
        cur.append(p.strip())
        cur_len += len(p) + 2
    if cur:
        chunks.append((cur_head, "\n\n".join(cur)))
    return chunks


def split_text(text: str, target_chars: int) -> list[tuple[str, str]]:
    """Return [(heading, chunk)] chunks, each <= target when possible."""
    text = (text or "").strip()
    if not text:
        return []
    target = max(500, int(target_chars))
    if len(text) <= target:
        return [("", text.strip())]

    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2 and len(text) > target:
        # pathological: no paragraph boundaries at all
        return [("", p) for p in _split_sentences(text, target)]

    head0 = ""
    for p in paragraphs:
        label = _heading_label(p)
        if label:
            head0 = label
            break

    # pre-label each paragraph with the heading it belongs to; heading
    # lines themselves travel in the chunk header, not the body
    labeled: list[tuple[str, str]] = []
    cur_head = head0
    for p in paragraphs:
        label = _heading_label(p)
        if label:
            cur_head = label
            continue
        labeled.append((cur_head, p))

    chunks = _pack(labeled, target, head0)
    # oversized single-paragraph chunks (rare): sentence-split as fallback
    final: list[tuple[str, str]] = []
    for head, chunk in chunks:
        if len(chunk) <= target * 1.5 or "\n\n" in chunk:
            final.append((head, chunk))
        else:
            final.extend((head, piece) for piece in _split_sentences(chunk, target))
    return final
