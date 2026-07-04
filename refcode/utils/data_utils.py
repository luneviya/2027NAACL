"""Small data helpers for reviewer smoke checks and lightweight scripts."""

import json


def read_jsonl(path):
    """Read a JSONL file into a list of dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_token_list(value):
    """Normalize a string or token list into a flat token list."""
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.extend(item.split())
            else:
                out.append(str(item))
        return out
    return str(value).split()
