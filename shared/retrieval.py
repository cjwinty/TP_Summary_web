import json
import logging
from typing import Optional

from database import (
    get_conn,
    get_entity_data,
    get_relations,
    _build_metadata_blob,
)

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
_SEARCH_LIMIT = 80


def _group_rows(rows: list, exclude_ids: set[int]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        rid = row[0]
        if rid in exclude_ids:
            continue
        if rid not in grouped:
            grouped[rid] = []
        grouped[rid].append({
            "chunk_text": row[1],
            "entity_type": row[2] or "Request",
            "chunk_type": row[3] or "comment",
            "distance": float(row[4]),
        })
    return grouped


def vector_search(
    query_embedding: list[float],
    max_entities: int = 10,
    chunk_char_limit: int = 1200,
    token_budget: int = 30000,
    exclude_ids: Optional[set[int]] = None,
    filter_clauses: Optional[list[str]] = None,
    filter_params: Optional[list] = None,
) -> tuple[str, list[dict]]:
    exclude_ids = exclude_ids or set()
    filter_clauses = filter_clauses or []
    filter_params = filter_params or []

    conn = get_conn()
    c = conn.cursor()

    if filter_clauses:
        where_sql = " AND ".join(filter_clauses)
        c.execute(
            "SELECT e.request_id, e.chunk_text, e.entity_type, e.chunk_type, "
            "e.embedding <=> %s::vector AS distance "
            "FROM embeddings e "
            "INNER JOIN entity_data ed ON e.request_id = ed.entity_id "
            "WHERE " + where_sql + " "
            "ORDER BY distance LIMIT %s",
            (json.dumps(query_embedding), *filter_params, _SEARCH_LIMIT),
        )
    else:
        c.execute(
            "SELECT e.request_id, e.chunk_text, e.entity_type, e.chunk_type, "
            "e.embedding <=> %s::vector AS distance "
            "FROM embeddings e ORDER BY distance LIMIT %s",
            (json.dumps(query_embedding), _SEARCH_LIMIT),
        )

    rows = c.fetchall()

    chunks_by_id = _group_rows(rows, exclude_ids)

    if not chunks_by_id and exclude_ids:
        chunks_by_id = _group_rows(rows, set())

    if not chunks_by_id:
        return "", []

    entity_order = sorted(
        chunks_by_id.keys(),
        key=lambda rid: min(c["distance"] for c in chunks_by_id[rid]),
    )

    selected = entity_order[:max_entities]
    char_budget = token_budget * CHARS_PER_TOKEN
    total_chars = 0
    context_parts = []
    sources = []

    for rid in selected:
        if total_chars >= char_budget:
            break

        chunks = chunks_by_id[rid]
        chunks.sort(key=lambda c: c["distance"])

        ed = get_entity_data(rid)
        if ed:
            et = ed.get("entity_type") or chunks[0]["entity_type"]
            state = ed.get("entity_state", "")
            profile = _build_metadata_blob(rid, et, ed)
            rels = get_relations(rid)
            if rels:
                rel_texts = []
                for rel in rels[:5]:
                    rt = rel.get("related_entity_type") or "?"
                    rn = rel.get("related_entity_name") or str(rel.get("related_entity_id", ""))
                    rel_texts.append(f"{rt} #{rn}")
                if profile:
                    profile += " | Related: " + ", ".join(rel_texts)
                else:
                    profile = f"[{et} #{rid}] | Related: " + ", ".join(rel_texts)
            entity_header = profile or f"[{et} #{rid}]"
            sources.append({"id": rid, "type": et, "state": state})
        else:
            et = chunks[0]["entity_type"]
            entity_header = f"[{et} #{rid}]"
            sources.append({"id": rid, "type": et, "state": ""})

        header_bytes = len(entity_header)
        if total_chars + header_bytes + 3 > char_budget:
            continue

        entity_lines = [entity_header]
        entity_chars = header_bytes

        type_chunks = [c for c in chunks if c["chunk_type"] in ("comment", "summary")]
        for cc in type_chunks:
            label = "Summary" if cc["chunk_type"] == "summary" else "Comment"
            text = cc["chunk_text"][:chunk_char_limit]
            line = f"  {label}: {text}"
            line_len = len(line)
            if entity_chars + 2 + line_len > char_budget - total_chars:
                break
            if len(type_chunks) > 1:
                entity_lines.append("")
                entity_chars += 1
            entity_lines.append(line)
            entity_chars += line_len

        if context_parts:
            context_parts.append("")
            total_chars += 1
        context_parts.extend(entity_lines)
        total_chars += entity_chars

    return "\n".join(context_parts), sources
