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


def index_policies(conn, policy_dir: Path = config.POLICY_DIR, embedder=None) -> int:
    """Read all markdown policies in policy_dir, chunk them, embed, and store in policy_chunks."""
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(config.EMBED_MODEL)

    chunks = [c for p in sorted(policy_dir.glob("*.md")) for c in chunk_policy(p)]
    if not chunks:
        return 0

    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    conn.execute("TRUNCATE policy_chunks")
    with conn.cursor() as cur:
        for c, v in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO policy_chunks (doc_id, title, text, embedding) VALUES (%s, %s, %s, %s)",
                (c["doc_id"], c["title"], c["text"], v),
            )
    return len(chunks)
