"""Index the internal policy manual (aml/policies/*.md) into pgvector."""
from __future__ import annotations

import re
from pathlib import Path

from . import config
from .schema import PolicyChunk


def chunk_policy(path: Path) -> list[PolicyChunk]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^# (.+)$", text, flags=re.M)
    if m is None:
        raise ValueError(f"{path.name}: missing '# ' title line")
    title = m.group(1).strip()
    doc_id = title.split(" ")[0]
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return [{"doc_id": doc_id, "title": title, "text": f"{title}\n## {p.strip()}"} for p in parts]


def index_policies(conn, policy_dir: Path = config.POLICY_DIR) -> int:
    raise NotImplementedError
