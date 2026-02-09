from __future__ import annotations

from pathlib import Path

from thought_vectors.data_loading import load_groups_from_path
from thought_vectors.preprocessing import normalize_apostrophes, normalize_text


def test_normalize_apostrophes_and_text_cleanup() -> None:
    text = "  It’s  John`s   book — really.\t\n"
    assert normalize_apostrophes(text).count("'") >= 2
    assert normalize_text(text) == "It's John's book - really."


def test_load_groups_from_csv_uses_first_column_and_preprocesses(tmp_path: Path) -> None:
    data = "text,label\nIt’s raining,weather\n  John`s cat ,animal\n"
    path = tmp_path / "data.csv"
    path.write_text(data, encoding="utf-8")

    groups = load_groups_from_path(path)

    assert groups == [["It's raining"], ["John's cat"]]
