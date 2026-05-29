import os
import psycopg2
import psycopg2.extras
import json
import logging
import threading
import re
import difflib
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5432")),
    "dbname": os.getenv("PG_DATABASE", "tp_query"),
    "user": os.getenv("PG_USER", "tp_query"),
    "password": os.getenv("PG_PASSWORD", "tp_query"),
}

_connection = None
_lock = threading.Lock()


def _get_conn():
    global _connection
    if _connection is None or _connection.closed:
        _connection = psycopg2.connect(**DB_CONFIG)
        _connection.autocommit = False
    _connection.rollback()
    return _connection


def _close_conn():
    global _connection
    if _connection:
        _connection.close()
        _connection = None


def init_db():
    conn = _get_conn()
    c = conn.cursor()

    # Try to create the vector extension in its own transaction
    # so a permissions failure doesn't prevent table creation
    try:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    except Exception:
        conn.rollback()
        logger.warning("Could not create vector extension (may need superuser). Tables will still be created.")

    # ── New entity_data table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS entity_data (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL DEFAULT 'Request',
            description TEXT,
            create_date TEXT,
            entity_state TEXT,
            entity_state_id INTEGER,
            project_id INTEGER,
            project_name TEXT DEFAULT '',
            client TEXT,
            product TEXT,
            release_version TEXT,
            site TEXT,
            custom_fields JSONB DEFAULT '{}',
            fetched_at TEXT
        )
    """)

    # ── Migrate: add custom field columns to entity_data ──
    for col, col_type in [
        ("customer_ref", "TEXT DEFAULT ''"),
        ("internal_priority", "TEXT DEFAULT ''"),
        ("support_level", "TEXT DEFAULT ''"),
        ("next_action", "TEXT DEFAULT ''"),
        ("paid_work", "TEXT DEFAULT ''"),
        ("downtime", "TEXT DEFAULT ''"),
        ("out_of_hours", "TEXT DEFAULT ''"),
        ("customer_chased_date", "TEXT DEFAULT ''"),
        ("stop_feedback_request", "TEXT DEFAULT ''"),
    ]:
        c.execute(f"ALTER TABLE entity_data ADD COLUMN IF NOT EXISTS {col} {col_type}")

    # ── New entity_relations table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS entity_relations (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER NOT NULL,
            entity_type TEXT NOT NULL DEFAULT 'Request',
            related_entity_id INTEGER NOT NULL,
            related_entity_type TEXT,
            related_entity_name TEXT,
            related_entity_state TEXT,
            relation_id INTEGER,
            fetched_at TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_entity_relations_entity ON entity_relations(entity_id)")

    # ── comments table (add entity_type if missing) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            request_id INTEGER PRIMARY KEY,
            comment_data JSONB,
            fetched_at TEXT,
            entity_type TEXT NOT NULL DEFAULT 'Request'
        )
    """)
    c.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='comments' AND column_name='entity_type'
    """)
    if not c.fetchone():
        c.execute("ALTER TABLE comments ADD COLUMN entity_type TEXT NOT NULL DEFAULT 'Request'")

    # ── summaries table (add entity_type if missing) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            request_id INTEGER PRIMARY KEY,
            summary_text TEXT,
            created_at TEXT,
            entity_type TEXT NOT NULL DEFAULT 'Request'
        )
    """)
    c.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='summaries' AND column_name='entity_type'
    """)
    if not c.fetchone():
        c.execute("ALTER TABLE summaries ADD COLUMN entity_type TEXT NOT NULL DEFAULT 'Request'")

    # ── Old request_custom_fields (migration target) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS request_custom_fields (
            request_id INTEGER PRIMARY KEY,
            client TEXT,
            product TEXT,
            release_version TEXT,
            site TEXT,
            fetched_at TEXT,
            metadata JSONB DEFAULT '{}'
        )
    """)

    # ── Migration: copy old request_custom_fields → entity_data ──
    c.execute("SELECT COUNT(*) FROM request_custom_fields")
    old_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM entity_data")
    new_count = c.fetchone()[0]
    if old_count > 0 and new_count == 0:
        c.execute("""
            INSERT INTO entity_data
                (entity_id, entity_type, client, product, release_version, site, fetched_at)
            SELECT request_id, 'Request', client, product, release_version, site, fetched_at
            FROM request_custom_fields
            ON CONFLICT (entity_id) DO NOTHING
        """)
        logger.info("Migrated %d records from request_custom_fields to entity_data", c.rowcount)

    # ── embeddings table (add entity_type if missing) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            request_id INTEGER NOT NULL REFERENCES comments(request_id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            embedding vector(1536),
            chunk_type TEXT DEFAULT 'comment',
            created_at TEXT,
            entity_type TEXT NOT NULL DEFAULT 'Request'
        )
    """)
    c.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='embeddings' AND column_name='entity_type'
    """)
    if not c.fetchone():
        c.execute("ALTER TABLE embeddings ADD COLUMN entity_type TEXT NOT NULL DEFAULT 'Request'")

    c.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            is_active BOOLEAN DEFAULT FALSE,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS prompt_chains (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS prompt_chain_steps (
            id SERIAL PRIMARY KEY,
            chain_id INTEGER NOT NULL REFERENCES prompt_chains(id) ON DELETE CASCADE,
            step_order INTEGER NOT NULL,
            name TEXT NOT NULL,
            step_type TEXT DEFAULT 'llm',
            prompt_template TEXT NOT NULL,
            input_variable TEXT DEFAULT '',
            output_variable TEXT DEFAULT '',
            variables JSONB DEFAULT '{}'
        )
    """)
    c.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='prompt_chain_steps' AND column_name='step_type'
    """)
    if not c.fetchone():
        c.execute("ALTER TABLE prompt_chain_steps ADD COLUMN step_type TEXT DEFAULT 'llm'")
    c.execute("""
        CREATE TABLE IF NOT EXISTS prompt_chain_runs (
            id SERIAL PRIMARY KEY,
            chain_id INTEGER NOT NULL REFERENCES prompt_chains(id) ON DELETE CASCADE,
            initial_input TEXT,
            status TEXT DEFAULT 'running',
            final_output TEXT,
            error TEXT,
            context JSONB DEFAULT '{}',
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS prompt_chain_run_steps (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES prompt_chain_runs(id) ON DELETE CASCADE,
            step_id INTEGER REFERENCES prompt_chain_steps(id),
            step_order INTEGER NOT NULL,
            name TEXT,
            input_sent TEXT,
            output TEXT,
            status TEXT DEFAULT 'pending',
            error TEXT,
            duration_ms INTEGER DEFAULT 0,
            executed_at TEXT
        )
    """)
    for col, col_type in [("step_id", "INTEGER"), ("error", "TEXT"), ("executed_at", "TEXT")]:
        c.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='prompt_chain_run_steps' AND column_name=%s",
            (col,),
        )
        if not c.fetchone():
            c.execute(f"ALTER TABLE prompt_chain_run_steps ADD COLUMN {col} {col_type}")
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_comments_fetched_at ON comments(fetched_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_custom_fields_client ON request_custom_fields(client)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_custom_fields_product ON request_custom_fields(product)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_custom_fields_release ON request_custom_fields(release_version)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_custom_fields_site ON request_custom_fields(site)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_request ON embeddings(request_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prompt_chains_name ON prompt_chains(name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_entity_data_type ON entity_data(entity_type)")
    # Add project_name column if missing
    try:
        c.execute("ALTER TABLE entity_data ADD COLUMN IF NOT EXISTS project_name TEXT DEFAULT ''")
    except Exception:
        try:
            c.execute("ALTER TABLE entity_data ADD COLUMN project_name TEXT DEFAULT ''")
        except Exception:
            pass
    c.execute("CREATE INDEX IF NOT EXISTS idx_entity_data_project ON entity_data(project_id)")
    conn.commit()

    # ── Migrate: rename prompt names from US to UK spelling ──
    for old_name, new_name in [
        ("summarize", "summarise"),
        ("summarize_search", "summarise_search"),
    ]:
        c.execute("UPDATE prompts SET name = %s WHERE name = %s", (new_name, old_name))
    conn.commit()

    init_default_prompts()
    logger.info("PostgreSQL schema initialised")


DEFAULT_PROMPTS = {
    "summarise": """You are a {entity_type} analyser. Read the EXISTING comments/conversation for {entity_type} #{entity_id} and produce a detailed summary.

For each {entity_type}, provide a structured summary with these sections:
- Issue: What was the problem or request about?
- Actions Taken: What steps were taken to resolve the issue?
- Current Status: Is it resolved, pending, or escalated?

Be detailed and specific. Use information only from the provided comments.

{entity_type} #{entity_id}:
""",
    "extract_issues": """Analyse the following {entity_type} comments and classify the issue into one or more of these categories:
- Bug Report
- Feature Request
- Performance Issue
- Data/Integration Issue
- User Training/Question
- Account/Access Issue
- Configuration Issue
- Other

Return only the category names, comma-separated.

Comments: {comments}
""",
    "refine_search": """Given the following search query, generate a list of related search terms that might help find relevant results.
Return one term per line.

Query: {query}
""",
    "summarise_search": """You analysed cached support data for the query: "{query}"

Found {match_count} matching entries. Here are the most relevant:

{results_text}

Provide a concise executive summary of what these results indicate about the query topic.""",
}


def init_default_prompts():
    defaults = DEFAULT_PROMPTS
    for name, content in defaults.items():
        with _lock:
            conn = _get_conn()
            c = conn.cursor()
            c.execute("SELECT 1 FROM prompts WHERE name = %s", (name,))
            if not c.fetchone():
                now = datetime.now().isoformat()
                c.execute(
                    "INSERT INTO prompts (name, content, is_active, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)",
                    (name, content, name == "summarise", now, now),
                )
                conn.commit()


def get_cached_comments(request_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT comment_data, fetched_at FROM comments WHERE request_id = %s", (request_id,))
        row = c.fetchone()
    if row:
        data = row["comment_data"]
        comment_data = json.loads(data) if isinstance(data, str) else data
        return comment_data, row["fetched_at"]
    return None, None


def save_comments(request_id, comments, entity_type="Request"):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO comments (request_id, comment_data, fetched_at, entity_type) VALUES (%s, %s, %s, %s) ON CONFLICT (request_id) DO UPDATE SET comment_data = excluded.comment_data, fetched_at = excluded.fetched_at, entity_type = excluded.entity_type",
            (request_id, json.dumps(comments), now, entity_type),
        )
        conn.commit()


def delete_comments(request_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM comments WHERE request_id = %s", (request_id,))
        conn.commit()


def delete_embeddings(request_id: int):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM embeddings WHERE request_id = %s", (request_id,))
        conn.commit()


def get_cached_entity_type(entity_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT entity_type FROM entity_data WHERE entity_id = %s", (entity_id,))
        row = c.fetchone()
    return row["entity_type"] if row else None


def save_entity_data(entity_id, entity_type, description=None, create_date=None, entity_state=None, entity_state_id=None, project_id=None, client="", product="", release_version="", site="", project_name="", customer_ref="", internal_priority="", support_level="", next_action="", paid_work="", downtime="", out_of_hours="", customer_chased_date="", stop_feedback_request="", custom_fields=None, fetched_at=None):
    if custom_fields is None:
        custom_fields = {}
    if fetched_at is None:
        fetched_at = datetime.now().isoformat()
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO entity_data (entity_id, entity_type, description, create_date, entity_state, entity_state_id, project_id, project_name, client, product, release_version, site, customer_ref, internal_priority, support_level, next_action, paid_work, downtime, out_of_hours, customer_chased_date, stop_feedback_request, custom_fields, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id) DO UPDATE SET
                entity_type = excluded.entity_type,
                description = excluded.description,
                create_date = excluded.create_date,
                entity_state = excluded.entity_state,
                entity_state_id = excluded.entity_state_id,
                project_id = excluded.project_id,
                project_name = excluded.project_name,
                client = excluded.client,
                product = excluded.product,
                release_version = excluded.release_version,
                site = excluded.site,
                customer_ref = excluded.customer_ref,
                internal_priority = excluded.internal_priority,
                support_level = excluded.support_level,
                next_action = excluded.next_action,
                paid_work = excluded.paid_work,
                downtime = excluded.downtime,
                out_of_hours = excluded.out_of_hours,
                customer_chased_date = excluded.customer_chased_date,
                stop_feedback_request = excluded.stop_feedback_request,
                custom_fields = excluded.custom_fields,
                fetched_at = excluded.fetched_at
        """, (entity_id, entity_type, description, create_date, entity_state, entity_state_id, project_id, project_name, client, product, release_version, site, customer_ref, internal_priority, support_level, next_action, paid_work, downtime, out_of_hours, customer_chased_date, stop_feedback_request, json.dumps(custom_fields), fetched_at))
        conn.commit()


def get_entity_data(entity_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT * FROM entity_data WHERE entity_id = %s", (entity_id,))
        row = c.fetchone()
    if row:
        row["custom_fields"] = json.loads(row["custom_fields"]) if isinstance(row.get("custom_fields"), str) else (row.get("custom_fields") or {})
        return dict(row)
    return None


def get_cached_projects():
    """Return distinct projects from entity_data, sorted by name."""
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT DISTINCT project_id, project_name FROM entity_data WHERE project_id IS NOT NULL ORDER BY project_name")
        return [{"project_id": r[0], "project_name": r[1] or f"Project {r[0]}"} for r in c.fetchall()]


def get_entity_types_for_project(project_id: int):
    """Return distinct entity_types cached for a given project."""
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT DISTINCT entity_type FROM entity_data WHERE project_id = %s ORDER BY entity_type", (project_id,))
        return [r[0] for r in c.fetchall()]


def get_entities_by_project_and_type(project_id: int, entity_type: str):
    """Return entity list for a project+type combo."""
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("""
            SELECT entity_id, entity_type, description, entity_state, create_date
            FROM entity_data
            WHERE project_id = %s AND entity_type = %s
            ORDER BY entity_id
        """, (project_id, entity_type))
        return [
            {"id": r["entity_id"], "entity_type": r["entity_type"],
             "description": (r["description"] or "")[:200],
             "state": r["entity_state"] or "", "create_date": r["create_date"] or ""}
            for r in c.fetchall()
        ]


def save_relations(entity_id, entity_type, relations, fetched_at=None):
    if fetched_at is None:
        fetched_at = datetime.now().isoformat()
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM entity_relations WHERE entity_id = %s", (entity_id,))
        for rel in relations:
            c.execute("""
                INSERT INTO entity_relations (entity_id, entity_type, related_entity_id, related_entity_type, related_entity_name, related_entity_state, relation_id, fetched_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                entity_id,
                entity_type,
                rel.get("related_entity_id"),
                rel.get("related_entity_type"),
                rel.get("related_entity_name"),
                rel.get("related_entity_state"),
                rel.get("relation_id"),
                fetched_at,
            ))
        conn.commit()


def get_relations(entity_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT * FROM entity_relations WHERE entity_id = %s ORDER BY relation_id", (entity_id,))
        return [dict(r) for r in c.fetchall()]


def save_custom_fields(request_id, fields):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        now = datetime.now().isoformat()
        client = fields.get("Client") or fields.get("client") or ""
        product = fields.get("Product") or fields.get("product") or ""
        release = fields.get("Release Version") or fields.get("release_version") or ""
        site = fields.get("Site") or fields.get("site") or ""
        c.execute(
            """INSERT INTO request_custom_fields (request_id, client, product, release_version, site, fetched_at, metadata)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (request_id) DO UPDATE SET client = excluded.client, product = excluded.product,
               release_version = excluded.release_version, site = excluded.site, fetched_at = excluded.fetched_at,
               metadata = excluded.metadata""",
            (request_id, client, product, release, site, now, json.dumps(fields)),
        )
        conn.commit()


def get_custom_fields(request_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute(
            "SELECT client, product, release_version, site, fetched_at FROM request_custom_fields WHERE request_id = %s",
            (request_id,),
        )
        row = c.fetchone()

    if not row:
        return None, None

    fields = {
        "Client": row["client"],
        "Product": row["product"],
        "Release Version": row["release_version"],
        "Site": row["site"],
    }
    fetched_at = row["fetched_at"]

    return fields, fetched_at


def get_all_custom_field_names():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT DISTINCT jsonb_object_keys(metadata) FROM request_custom_fields WHERE metadata != '{}'::jsonb")
        keys = set()
        for row in c.fetchall():
            keys.add(row[0])
        return sorted(keys)


def get_custom_field_values(field_name):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute(
            "SELECT DISTINCT metadata->>%s FROM request_custom_fields WHERE metadata ? %s ORDER BY 1",
            (field_name, field_name),
        )
        return [row[0] for row in c.fetchall() if row[0]]


def save_summary(request_id, summary_text, entity_type="Request"):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO summaries (request_id, summary_text, created_at, entity_type) VALUES (%s, %s, %s, %s) ON CONFLICT (request_id) DO UPDATE SET summary_text = excluded.summary_text, created_at = excluded.created_at, entity_type = excluded.entity_type",
            (request_id, summary_text, now, entity_type),
        )
        conn.commit()


def get_summary(request_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT summary_text, created_at FROM summaries WHERE request_id = %s", (request_id,))
        row = c.fetchone()
    if row:
        return row[0], row[1]
    return None, None


def get_summary_with_cache_time(request_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute(
            "SELECT s.summary_text as summary, s.created_at, c.fetched_at FROM summaries s LEFT JOIN comments c ON c.request_id = s.request_id WHERE s.request_id = %s",
            (request_id,),
        )
        row = c.fetchone()
    if row:
        return {"summary": row["summary"], "created_at": row["created_at"], "fetched_at": row["fetched_at"]}
    return None


def get_all_summaries():
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT request_id as id, entity_type, created_at as created FROM summaries ORDER BY request_id DESC")
        return [{"id": r["id"], "entity_type": r["entity_type"], "created": r["created"]} for r in c.fetchall()]


def get_summaries_page(limit=50, offset=0):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT request_id as id, entity_type, created_at as created FROM summaries ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset))
        return [{"id": r["id"], "entity_type": r["entity_type"], "created": r["created"]} for r in c.fetchall()]


def get_summary_count():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM summaries")
        return c.fetchone()[0]


def delete_summary(request_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM summaries WHERE request_id = %s", (request_id,))
        conn.commit()


def delete_all_summaries():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM summaries")
        conn.commit()


def clear_entity_data():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM entity_data")
        conn.commit()


def clear_entity_relations():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM entity_relations")
        conn.commit()


def clear_all_chat_history():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM chat_history")
        conn.commit()


def clear_all_cached_data():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM entity_relations")
        c.execute("DELETE FROM entity_data")
        c.execute("DELETE FROM embeddings")
        c.execute("DELETE FROM summaries")
        c.execute("DELETE FROM request_custom_fields")
        c.execute("DELETE FROM comments")
        c.execute("DELETE FROM chat_history")
        conn.commit()


def get_all_cached_ids():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT request_id FROM comments ORDER BY request_id")
        return [row[0] for row in c.fetchall()]


def get_cache_counts():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM comments"); comments = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM summaries"); summaries = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM request_custom_fields"); custom = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM embeddings"); embeddings = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM entity_data"); entity_data_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM entity_relations"); entity_relations_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM chat_history"); chat_history = c.fetchone()[0]
    return {
        "comments": comments, "summaries": summaries,
        "custom_fields": custom, "embeddings": embeddings,
        "entity_data": entity_data_count,
        "entity_relations": entity_relations_count,
        "chat_history": chat_history,
    }


def get_max_min_request_id():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("SELECT MIN(request_id), MAX(request_id) FROM comments")
        row = c.fetchone()
    return {"min": row[0], "max": row[1]}


def search_cached_comments(query, custom_field_filter=None, date_filter=None, limit=1000):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        params = [f"%{query}%"]
        sql = "SELECT c.request_id, c.comment_data, c.fetched_at, c.entity_type, cf.client, cf.product FROM comments c LEFT JOIN request_custom_fields cf ON c.request_id = cf.request_id WHERE c.comment_data::text ILIKE %s"

        if custom_field_filter:
            if isinstance(custom_field_filter, dict) and "filters" in custom_field_filter:
                logic = custom_field_filter.get("logic", "AND")
                clauses = []
                for f in custom_field_filter["filters"]:
                    clauses.append("cf.%s ILIKE %s" % (f["field_name"], "%%s%%"))
                    params.append(f["field_value"])
                sql += " AND (" + (" %s " % logic).join(clauses) + ")"
            else:
                sql += " AND cf.%s ILIKE %s" % (custom_field_filter["field_name"], "%%%s%%")
                params.append(custom_field_filter["field_value"])

        if date_filter:
            if date_filter.get("start_date"):
                sql += " AND c.fetched_at >= %s"
                params.append(date_filter["start_date"])
            if date_filter.get("end_date"):
                sql += " AND c.fetched_at <= %s"
                params.append(date_filter["end_date"])

        sql += " LIMIT %s"
        params.append(limit)
        c.execute(sql, params)
        results = []
        for row in c.fetchall():
            comment_data = json.loads(row["comment_data"]) if isinstance(row["comment_data"], str) else row["comment_data"]
            for comment in (comment_data or []):
                text = comment.get("text", "")
                if query.lower() in text.lower():
                    results.append({
                        "request_id": row["request_id"],
                        "text": text[:500],
                        "source": "comments",
                        "entity_type": row.get("entity_type") or "Request",
                    })
        return results


def search_summaries(query, custom_field_filter=None, date_filter=None, limit=1000):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        params = [f"%{query}%"]
        sql = "SELECT s.request_id, s.summary_text, s.created_at, s.entity_type FROM summaries s WHERE s.summary_text ILIKE %s"

        if date_filter:
            if date_filter.get("start_date"):
                sql += " AND s.created_at >= %s"
                params.append(date_filter["start_date"])
            if date_filter.get("end_date"):
                sql += " AND s.created_at <= %s"
                params.append(date_filter["end_date"])

        sql += " LIMIT %s"
        params.append(limit)
        c.execute(sql, params)
        return [{"request_id": r["request_id"], "text": r["summary_text"][:500], "source": "summaries", "entity_type": r.get("entity_type") or "Request"} for r in c.fetchall()]


def search_and_fetch_full(query, custom_field_filter=None, date_filter=None, limit=200):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        params = [f"%{query}%"]
        sql = """SELECT c.request_id, c.comment_data, c.fetched_at, c.entity_type,
                        cf.client, cf.product, cf.release_version, cf.site,
                        s.summary_text as summary
                 FROM comments c
                 LEFT JOIN request_custom_fields cf ON c.request_id = cf.request_id
                 LEFT JOIN summaries s ON c.request_id = s.request_id
                 WHERE c.comment_data::text ILIKE %s"""

        if custom_field_filter:
            if isinstance(custom_field_filter, dict) and "filters" in custom_field_filter:
                logic = custom_field_filter.get("logic", "AND")
                clauses = []
                for f in custom_field_filter["filters"]:
                    safe_field = re.sub(r'[^a-z_]', '', f["field_name"].lower().replace(" ", "_"))
                    clauses.append(f"cf.{safe_field} ILIKE %s")
                    params.append(f"%{f['field_value']}%")
                sql += " AND (" + (" %s " % logic).join(clauses) + ")"
            else:
                safe_field = re.sub(r'[^a-z_]', '', custom_field_filter["field_name"].lower().replace(" ", "_"))
                sql += f" AND cf.{safe_field} ILIKE %s"
                params.append(f"%{custom_field_filter['field_value']}%")

        sql += " LIMIT %s"
        params.append(limit)
        c.execute(sql, params)
        results = []
        for row in c.fetchall():
            comment_data = json.loads(row["comment_data"]) if isinstance(row["comment_data"], str) else row["comment_data"]
            if comment_data:
                results.append({
                    "request_id": row["request_id"],
                    "entity_type": row.get("entity_type") or "Request",
                    "client": row.get("client") or "",
                    "product": row.get("product") or "",
                    "site": row.get("site") or "",
                    "release_version": row.get("release_version") or "",
                    "comments": comment_data,
                    "summary": row.get("summary") or "",
                    "match_score": 1.0,
                })
        return results


def check_database_health():
    with _lock:
        try:
            conn = _get_conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM comments"); comments = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM summaries"); summaries = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM request_custom_fields"); custom = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM embeddings"); embeddings_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM entity_data"); entity_data_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM entity_relations"); entity_relations_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM chat_history"); chat_history = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM request_custom_fields cf WHERE cf.request_id NOT IN (SELECT request_id FROM comments)"); orphan = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM embeddings e WHERE e.request_id NOT IN (SELECT request_id FROM comments)"); orphan_emb = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM entity_data e WHERE e.entity_id NOT IN (SELECT request_id FROM comments)"); orphan_ed = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM entity_relations r WHERE r.entity_id NOT IN (SELECT request_id FROM comments)"); orphan_er = c.fetchone()[0]
            c.execute("SELECT pg_database_size(current_database())"); size = c.fetchone()[0]

            # Index listing
            c.execute("""SELECT schemaname, tablename, indexname
                         FROM pg_indexes
                         WHERE tablename IN ('comments','summaries','request_custom_fields','embeddings','entity_data','entity_relations','chat_history')
                         ORDER BY tablename, indexname""")
            indexes = [{"table": r[1], "name": r[2]} for r in c.fetchall()]

            msgs = []
            if orphan > 0:
                msgs.append(f"{orphan} orphan custom field records")
            if orphan_emb > 0:
                msgs.append(f"{orphan_emb} orphan embedding records")
            if orphan_ed > 0:
                msgs.append(f"{orphan_ed} orphan entity_data records")
            if orphan_er > 0:
                msgs.append(f"{orphan_er} orphan entity_relations records")

            status = "healthy"
            if orphan > 0 or orphan_emb > 0 or orphan_ed > 0 or orphan_er > 0:
                status = "warning"

            return {
                "status": status,
                "db_size_mb": size / (1024 * 1024),
                "row_counts": {
                    "comments": comments,
                    "summaries": summaries,
                    "request_custom_fields": custom,
                    "embeddings": embeddings_count,
                    "entity_data": entity_data_count,
                    "entity_relations": entity_relations_count,
                    "chat_history": chat_history,
                },
                "orphan_fields": orphan,
                "orphan_percentage": round(orphan / max(custom, 1) * 100, 1) if custom > 0 else 0,
                "orphan_embeddings": orphan_emb,
                "orphan_embedding_percentage": round(orphan_emb / max(embeddings_count, 1) * 100, 1),
                "orphan_entity_data": orphan_ed,
                "orphan_entity_relations": orphan_er,
                "index_count": len(indexes),
                "indexes": indexes,
                "messages": msgs if msgs else ["Database is healthy."],
            }
        except Exception as e:
            return {"status": "error", "db_size_mb": 0, "row_counts": {}, "orphan_fields": 0, "orphan_percentage": 0, "orphan_embeddings": 0, "orphan_embedding_percentage": 0, "index_count": 0, "indexes": [], "messages": [str(e)]}


def optimise_database():
    with _lock:
        try:
            conn = _get_conn()
            c = conn.cursor()

            # Clean orphans in a normal transaction
            c.execute("DELETE FROM request_custom_fields cf WHERE cf.request_id NOT IN (SELECT request_id FROM comments)")
            del_fields = c.rowcount
            c.execute("DELETE FROM summaries WHERE request_id NOT IN (SELECT request_id FROM comments)")
            del_summaries = c.rowcount
            c.execute("DELETE FROM embeddings WHERE request_id NOT IN (SELECT request_id FROM comments)")
            del_embeddings = c.rowcount
            c.execute("DELETE FROM entity_data e WHERE e.entity_id NOT IN (SELECT request_id FROM comments)")
            del_entity_data = c.rowcount
            c.execute("DELETE FROM entity_relations r WHERE r.entity_id NOT IN (SELECT request_id FROM comments)")
            del_entity_relations = c.rowcount
            conn.commit()

            # VACUUM ANALYZE must run outside any transaction block
            old_autocommit = conn.autocommit
            conn.autocommit = True
            c.execute("VACUUM ANALYZE")
            conn.autocommit = old_autocommit

            parts = []
            if del_fields: parts.append(f"custom fields: {del_fields}")
            if del_summaries: parts.append(f"summaries: {del_summaries}")
            if del_embeddings: parts.append(f"embeddings: {del_embeddings}")
            if del_entity_data: parts.append(f"entity_data: {del_entity_data}")
            if del_entity_relations: parts.append(f"entity_relations: {del_entity_relations}")
            detail = f" ({', '.join(parts)})" if parts else ""
            return {
                "message": f"VACUUM ANALYZE + orphan cleanup{detail}",
                "deleted_orphans": del_fields,
                "deleted_summaries": del_summaries,
                "deleted_embeddings": del_embeddings,
                "deleted_entity_data": del_entity_data,
                "deleted_entity_relations": del_entity_relations,
            }
        except Exception as e:
            return {"message": f"Optimisation failed: {e}", "deleted_orphans": 0, "deleted_summaries": 0, "deleted_embeddings": 0}


def analyse_indexes():
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("""SELECT schemaname, tablename, indexname, indexdef
                     FROM pg_indexes
                     WHERE tablename IN ('comments','summaries','request_custom_fields','embeddings','entity_data','entity_relations','chat_history')
                     ORDER BY tablename, indexname""")
        return [{"schema": r[0], "table": r[1], "index": r[2], "definition": r[3]} for r in c.fetchall()]


def save_prompt(name, content, is_active=False):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO prompts (name, content, is_active, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (name) DO UPDATE SET content = excluded.content, is_active = excluded.is_active, updated_at = excluded.updated_at",
            (name, content, is_active, now, now),
        )
        conn.commit()


def get_prompt(name):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT content, is_active FROM prompts WHERE name = %s", (name,))
        row = c.fetchone()
    if row:
        return {"content": row["content"], "is_active": row["is_active"]}
    return None


def get_all_prompts():
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT id, name, content, is_active, created_at, updated_at FROM prompts ORDER BY name")
        return [dict(r) for r in c.fetchall()]


def get_conn():
    return _get_conn()


def _ensure_chat_history_table():
    conn = _get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT
        )
    """)
    conn.commit()


def save_chat_message(session_id: str, role: str, content: str):
    _ensure_chat_history_table()
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO chat_history (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
            (session_id, role, content, now),
        )
        conn.commit()


def get_chat_history(session_id: str, limit: int = 12) -> list[dict]:
    _ensure_chat_history_table()
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute(
            "SELECT role, content, created_at FROM chat_history WHERE session_id = %s ORDER BY created_at DESC",
            (session_id,),
        )
        rows = c.fetchall()
    history = [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]
    history.reverse()
    return history[-limit:]


def clear_chat_history(session_id: str):
    _ensure_chat_history_table()
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
        conn.commit()


def _build_embedding_prefix(entity_id: int, entity_type: str, entity_data: dict | None = None) -> str:
    if entity_data is None:
        entity_data = {}
    fields = [f"{entity_type} #{entity_id}"]
    state = entity_data.get("entity_state") or ""
    if state:
        fields.append(f"State: {state}")
    client = entity_data.get("client") or ""
    if client:
        fields.append(f"Client: {client}")
    product = entity_data.get("product") or ""
    if product:
        fields.append(f"Product: {product}")
    return f"[{' | '.join(fields)}]"


def _build_metadata_blob(entity_id: int, entity_type: str, entity_data: dict | None = None) -> str | None:
    if not entity_data:
        return None
    parts = [f"[{entity_type} #{entity_id}]"]
    state = entity_data.get("entity_state") or ""
    if state:
        parts.append(f"State: {state}")
    pname = entity_data.get("project_name") or ""
    if pname:
        parts.append(f"Project: {pname}")
    client = entity_data.get("client") or ""
    if client:
        parts.append(f"Client: {client}")
    product = entity_data.get("product") or ""
    if product:
        parts.append(f"Product: {product}")
    version = entity_data.get("release_version") or ""
    if version:
        parts.append(f"Version: {version}")
    cf = entity_data.get("custom_fields") or {}
    skip_keys = {"Client", "client", "Product", "product", "Release Version", "release_version", "Site", "site"}
    for key, val in cf.items():
        if val and key not in skip_keys:
            parts.append(f"{key}: {val}")
    desc = entity_data.get("description") or ""
    if desc and desc != "true":
        parts.append(f"Description: {desc[:2000]}")
    return " | ".join(parts)


def auto_index_request_web(request_id: int, index_summary: bool = True) -> int:
    count = 0
    try:
        from shared import config as cfg
        cfg.initialise_llm()
        from shared.llm_providers import LLMClient
    except Exception:
        return 0

    entity_type = get_cached_entity_type(request_id) or "Request"

    entity_data = get_entity_data(request_id)
    prefix = _build_embedding_prefix(request_id, entity_type, entity_data)

    # Delete existing embeddings so we always write fresh
    conn = _get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM embeddings WHERE request_id = %s", (request_id,))
    conn.commit()

    comments, _ = get_cached_comments(request_id)
    if comments:
        for comment in (comments if isinstance(comments, list) else []):
            text = comment.get("text", "").strip() if isinstance(comment, dict) else ""
            if not text or len(text) < 20:
                continue
            try:
                prefixed = f"{prefix} {text[:5000]}"
                embedding = LLMClient.generate_embedding(prefixed)
                if embedding and any(v != 0.0 for v in embedding):
                    c.execute(
                        "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                        (request_id, prefixed, json.dumps(embedding), "comment", datetime.now().isoformat(), entity_type),
                    )
                    conn.commit()
                    count += 1
            except Exception:
                continue

    if index_summary:
        summary, _ = get_summary(request_id)
        if summary and summary.strip() and len(summary) >= 20:
            try:
                prefixed = f"{prefix} {summary[:5000]}"
                embedding = LLMClient.generate_embedding(prefixed)
                if embedding and any(v != 0.0 for v in embedding):
                    c.execute(
                        "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                        (request_id, prefixed, json.dumps(embedding), "summary", datetime.now().isoformat(), entity_type),
                    )
                    conn.commit()
                    count += 1
            except Exception:
                pass

    # Metadata blob
    blob = _build_metadata_blob(request_id, entity_type, entity_data)
    if blob:
        try:
            embedding = LLMClient.generate_embedding(blob[:5000])
            if embedding and any(v != 0.0 for v in embedding):
                c.execute(
                    "INSERT INTO embeddings (request_id, chunk_text, embedding, chunk_type, created_at, entity_type) VALUES (%s, %s, %s::vector, %s, %s, %s)",
                    (request_id, blob[:5000], json.dumps(embedding), "metadata", datetime.now().isoformat(), entity_type),
                )
                conn.commit()
                count += 1
        except Exception:
            pass

    return count


def search_cached_issues_by_product_keyword(product: str | None, keywords: list[str], limit: int = 10) -> list[dict]:
    if not keywords:
        return []

    keywords = [k.strip().lower() for k in keywords if k.strip()]
    if not keywords:
        return []

    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        like_clauses = []
        params = []
        for kw in keywords:
            escaped = kw.replace('%', '\\%').replace('_', '\\_')
            like_clauses.append("c.comment_data::text ILIKE %s ESCAPE '\\'")
            params.append(f'%{escaped}%')

        where_clause = ' OR '.join(like_clauses)

        if product and product.strip() and product.lower() != "not recorded":
            product_filter = " AND LOWER(rcf.product) = %s"
            params.append(product.strip().lower())
        else:
            product_filter = ""

        sql = f"""
            SELECT DISTINCT c.request_id, c.comment_data, rcf.product
            FROM comments c
            LEFT JOIN request_custom_fields rcf ON c.request_id = rcf.request_id
            WHERE ({where_clause}){product_filter}
        """

        try:
            c.execute(sql, params)
            all_rows = c.fetchall()
        except Exception:
            c.execute("""
                SELECT c.request_id, c.comment_data, rcf.product
                FROM comments c
                LEFT JOIN request_custom_fields rcf ON c.request_id = rcf.request_id
            """)
            all_rows = c.fetchall()
            if product and product.strip() and product.lower() != "not recorded":
                product_lower = product.strip().lower()
                all_rows = [r for r in all_rows if r["product"] and r["product"].lower() == product_lower]

    request_scores = {}

    for row in all_rows:
        req_id = row["request_id"]
        comment_json = row["comment_data"]
        prod = row["product"]

        try:
            comment_data = json.loads(comment_json) if isinstance(comment_json, str) else (comment_json or [])
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(comment_data, list):
            comment_data = [{"text": str(comment_data)}]

        for entry in comment_data:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text", "")
            if not text:
                continue

            text_lower = text.lower()

            matched_keywords = []
            total_score = 0

            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, text_lower):
                    matched_keywords.append(kw)
                    total_score += 3
                    continue

                if kw in text_lower:
                    matched_keywords.append(kw)
                    total_score += 2
                    continue

                words = re.findall(r'\b\w+\b', text_lower)
                if words:
                    close = difflib.get_close_matches(kw, words, n=1, cutoff=0.8)
                    if close:
                        matched_keywords.append(kw)
                        total_score += 1

            if matched_keywords:
                existing = request_scores.get(req_id)
                if not existing or total_score > existing["_score"]:
                    request_scores[req_id] = {
                        "request_id": req_id,
                        "text": text[:2000],
                        "match_reason": f"Keywords '{', '.join(matched_keywords)}' found in comments",
                        "product": prod or "Unknown",
                        "_score": total_score,
                    }

    results = sorted(request_scores.values(), key=lambda x: x["_score"], reverse=True)
    return results[:limit]


def save_chain(name, description="", steps=None):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        now = datetime.now().isoformat()
        c.execute("INSERT INTO prompt_chains (name, description, created_at, updated_at) VALUES (%s, %s, %s, %s) RETURNING id",
                  (name, description, now, now))
        chain_id = c.fetchone()["id"]
        if steps:
            for s in steps:
                c.execute(
                    "INSERT INTO prompt_chain_steps (chain_id, step_order, name, prompt_template, input_variable, output_variable, variables) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (chain_id, s["step_order"], s["name"], s["prompt_template"], s.get("input_variable", ""), s.get("output_variable", ""), json.dumps(s.get("variables", {}))),
                )
        conn.commit()
        return chain_id


def get_chain(chain_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT * FROM prompt_chains WHERE id = %s", (chain_id,))
        chain = c.fetchone()
        if not chain:
            return None
        c.execute("SELECT * FROM prompt_chain_steps WHERE chain_id = %s ORDER BY step_order", (chain_id,))
        chain["steps"] = [dict(r) for r in c.fetchall()]
        return dict(chain)


def list_chains():
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT id, name, description, created_at, updated_at FROM prompt_chains ORDER BY name")
        return [dict(r) for r in c.fetchall()]


def update_chain(chain_id, name, description="", steps=None):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute("UPDATE prompt_chains SET name = %s, description = %s, updated_at = %s WHERE id = %s",
                  (name, description, now, chain_id))
        c.execute("DELETE FROM prompt_chain_steps WHERE chain_id = %s", (chain_id,))
        if steps:
            for s in steps:
                c.execute(
                    "INSERT INTO prompt_chain_steps (chain_id, step_order, name, prompt_template, input_variable, output_variable, variables) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (chain_id, s["step_order"], s["name"], s["prompt_template"], s.get("input_variable", ""), s.get("output_variable", ""), json.dumps(s.get("variables", {}))),
                )
        conn.commit()


def delete_chain(chain_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM prompt_chains WHERE id = %s", (chain_id,))
        conn.commit()


def create_run(chain_id, initial_input):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO prompt_chain_runs (chain_id, initial_input, status, created_at) VALUES (%s, %s, 'running', %s) RETURNING id",
            (chain_id, initial_input, now),
        )
        run_id = c.fetchone()["id"]
        conn.commit()
        return run_id


def update_run_step(run_id, step_id=None, step_order=0, input_sent="", output_received=None, status="completed", error=None, duration_ms=0, name=None):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute(
            "INSERT INTO prompt_chain_run_steps (run_id, step_id, step_order, name, input_sent, output, status, error, duration_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (run_id, step_id, step_order, name, input_sent, output_received, status, error, duration_ms),
        )
        conn.commit()


def finish_run(run_id, status, final_output="", error="", context=None):
    with _lock:
        conn = _get_conn()
        c = conn.cursor()
        c.execute(
            "UPDATE prompt_chain_runs SET status = %s, final_output = %s, error = %s, context = %s WHERE id = %s",
            (status, final_output, error, json.dumps(context or {}), run_id),
        )
        conn.commit()


def get_run(run_id):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT * FROM prompt_chain_runs WHERE id = %s", (run_id,))
        run = c.fetchone()
        if not run:
            return None
        c.execute("SELECT * FROM prompt_chain_run_steps WHERE run_id = %s ORDER BY step_order", (run_id,))
        run["steps"] = [dict(r) for r in c.fetchall()]
        return dict(run)


def list_runs(chain_id, limit=50):
    with _lock:
        conn = _get_conn()
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute(
            "SELECT id, chain_id, status, created_at, final_output FROM prompt_chain_runs WHERE chain_id = %s ORDER BY created_at DESC LIMIT %s",
            (chain_id, limit),
        )
        return [dict(r) for r in c.fetchall()]
