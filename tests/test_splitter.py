"""Tests for the section-aware splitter and chunked summarization."""
from __future__ import annotations

from app.splitter import split_text


def test_short_text_single_part():
    assert split_text("short text", 1500) == [("", "short text")]


def test_empty_text_no_parts():
    assert split_text("", 1500) == []
    assert split_text("   ", 1500) == []


def test_headings_label_chunks_not_body():
    long = "\n\n".join(f"# Head {i}\n\n" + ("para " * 200) for i in range(1, 6))
    parts = split_text(long, 1500)
    assert [h for h, _ in parts] == ["Head 1", "Head 2", "Head 3", "Head 4", "Head 5"]
    assert not any("# Head" in c for _, c in parts)  # heading lives in the header
    assert all(len(c) <= 1500 + 1300 for _, c in parts)  # paragraph floor


def test_paragraph_never_split():
    big_para = "word " * 300  # 1500 chars, one paragraph, no headings
    text = "\n\n".join([big_para] * 3)
    parts = split_text(text, 2000)
    assert all(p.strip() in [pp.strip() for pp in text.split("\n\n")] for _, p in parts)


def test_no_paragraph_boundaries_falls_back_to_sentences():
    text = "Sentence one. " * 800  # ~11k chars, zero \n\n
    parts = split_text(text, 1500)
    assert len(parts) >= 5
    assert all(h == "" for h, _ in parts)
    assert "".join(p for _, p in parts).count("Sentence") == 800


def test_leading_prose_gets_following_heading():
    text = "Intro paragraph before any heading.\n\n# Real Heading\n\n" + "body " * 400
    parts = split_text(text, 1500)
    assert parts[0][0] == "Real Heading"
    assert parts[0][1].startswith("A Conceptual") is False
    assert "Intro paragraph" in parts[0][1]


def test_pilcrow_stripped_from_heading_label():
    text = "# Event Loop¶\n\n" + "body " * 800
    parts = split_text(text, 1500)
    assert parts[0][0] == "Event Loop"


def test_body_on_line_after_heading_is_kept():
    # `## H\nbody` (no blank line) is common markdown: the body must not be
    # dropped along with the heading line.
    text = ("## Intro\nFACT_A lives right under the heading.\n\n" + "filler para. " * 60
            + "\n\n## Pricing\nFACT_B costs $42.\n\n" + "more filler. " * 60)
    parts = split_text(text, 600)
    joined = "\n".join(c for _, c in parts)
    assert "FACT_A" in joined and "FACT_B" in joined
    assert not any("## " in c for _, c in parts)  # heading line itself still stripped
    assert ("Pricing", "FACT_B costs $42.") in [(h, c.split("\n\n")[0]) for h, c in parts]
