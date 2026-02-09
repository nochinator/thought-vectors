from __future__ import annotations

import csv
import json
from pathlib import Path

from thought_vectors.preprocessing import normalize_text


def _clean(text: str, preprocess: bool) -> str:
    return normalize_text(text) if preprocess else text


def load_groups_from_path(path: Path, preprocess: bool = True) -> list[list[str]]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a top-level list.")

        groups: list[list[str]] = []
        for group in data:
            if isinstance(group, list):
                cleaned = [_clean(str(x), preprocess) for x in group]
                cleaned = [x for x in cleaned if x]
                if cleaned:
                    groups.append(cleaned)
            else:
                text = _clean(str(group), preprocess)
                if text:
                    groups.append([text])
        return groups

    if suffix == ".csv":
        groups: list[list[str]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if not row:
                    continue
                raw = str(row[0])
                if first and raw.strip().lower() in {"text", "sentence", "content"}:
                    first = False
                    continue
                first = False
                text = _clean(raw, preprocess)
                if text:
                    groups.append([text])
        return groups

    # default: jsonl with {"texts": [...]}
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        texts = obj.get("texts", [])
        cleaned = [_clean(str(x), preprocess) for x in texts]
        cleaned = [x for x in cleaned if x]
        if cleaned:
            groups.append(cleaned)
    return groups
