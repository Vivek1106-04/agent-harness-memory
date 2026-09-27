from __future__ import annotations

from harness.chunking import chunk_text


def test_sentences_become_separate_chunks() -> None:
    text = "Filler one. The access code for site_1_a is 4271. Filler two!"
    assert chunk_text(text) == [
        "Filler one.",
        "The access code for site_1_a is 4271.",
        "Filler two!",
    ]


def test_a_listing_becomes_one_chunk_per_line() -> None:
    assert chunk_text("doc_00.txt\ndoc_01.txt\n") == ["doc_00.txt", "doc_01.txt"]


def test_text_without_a_boundary_is_one_chunk() -> None:
    assert chunk_text("4271") == ["4271"]


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("  \n ") == []
