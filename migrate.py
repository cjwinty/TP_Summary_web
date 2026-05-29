"""
migrate.py — One-time SQLite → PostgreSQL migration script.

Usage:
    python migrate.py                    # Run migration
    python migrate.py --dry-run          # Preview without writing
    python migrate.py --sqlite path.db   # Use custom SQLite path

Requires PostgreSQL to be running with the schema already created
(run init_db() from database.py first, or let this script create tables).
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sqlite3
import psycopg2
import psycopg2.extras


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SQLITE_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "tp_cache.db")

PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5432")),
    "dbname": os.getenv("PG_DATABASE", "tp_query"),
    "user": os.getenv("PG_USER", "tp_query"),
    "password": os.getenv("PG_PASSWORD", "tp_query"),
}

BATCH_SIZE = 500


def sanitize(val):
    """Strip null bytes and null escape sequences from text."""
    if isinstance(val, str):
        val = val.replace("\x00", "")
        val = val.replace("\\u0000", "")
        return val
    return val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def connect_pg():
    conn = psycopg2.connect(**PG_CONFIG)
    conn.autocommit = False
    return conn


def create_pg_tables(conn):
    c = conn.cursor()
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    c.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            request_id INTEGER PRIMARY KEY,
            comment_data JSONB,
            fetched_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            request_id INTEGER PRIMARY KEY,
            summary_text TEXT,
            created_at TEXT
        )
    """)
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
    conn.commit()
    print("  [OK] PostgreSQL tables created")


def table_exists(cursor, table_name):
    cursor.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
        (table_name,),
    )
    return cursor.fetchone()[0]


# ---------------------------------------------------------------------------
# Table migrators
# ---------------------------------------------------------------------------

def copy_comments(sqlite, pg, dry_run):
    sq = sqlite.cursor()
    sq.execute("SELECT request_id, comment_data, fetched_at FROM comments ORDER BY request_id")
    rows = sq.fetchall()
    if not rows:
        print("  - comments: 0 rows (no data)")
        return {"comments": 0}
    count = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        if not dry_run:
            c = pg.cursor()
            psycopg2.extras.execute_values(
                c,
                "INSERT INTO comments (request_id, comment_data, fetched_at) VALUES %s ON CONFLICT (request_id) DO NOTHING",
                [(r[0], sanitize(r[1]) if isinstance(r[1], str) else json.dumps(r[1]), r[2]) for r in batch],
            )
            pg.commit()
        count += len(batch)
    print(f"  [OK] comments: {count} rows{' (dry run)' if dry_run else ''}")
    return {"comments": count}


def copy_summaries(sqlite, pg, dry_run):
    sq = sqlite.cursor()
    sq.execute("SELECT request_id, summary_text, created_at FROM summaries ORDER BY request_id")
    rows = sq.fetchall()
    if not rows:
        print("  - summaries: 0 rows (no data)")
        return {"summaries": 0}
    count = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        batch = [tuple(sanitize(v) for v in r) for r in batch]
        if not dry_run:
            c = pg.cursor()
            psycopg2.extras.execute_values(
                c,
                "INSERT INTO summaries (request_id, summary_text, created_at) VALUES %s ON CONFLICT (request_id) DO NOTHING",
                batch,
            )
            pg.commit()
        count += len(batch)
    print(f"  [OK] summaries: {count} rows{' (dry run)' if dry_run else ''}")
    return {"summaries": count}


def copy_custom_fields(sqlite, pg, dry_run):
    sq = sqlite.cursor()
    sq.execute("PRAGMA table_info(request_custom_fields)")
    cols = [row[1] for row in sq.fetchall()]
    has_metadata = "metadata" in cols

    sq.execute("SELECT * FROM request_custom_fields ORDER BY request_id")
    rows = sq.fetchall()
    if not rows:
        print("  - request_custom_fields: 0 rows (no data)")
        return {"custom_fields": 0}
    count = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        if not dry_run:
            pg_c = pg.cursor()
            for r in batch:
                rid = r[0]
                client = r[cols.index("client")] if "client" in cols else ""
                product = r[cols.index("product")] if "product" in cols else ""
                release_v = r[cols.index("release_version")] if "release_version" in cols else ""
                site = r[cols.index("site")] if "site" in cols else ""
                fetched = r[cols.index("fetched_at")] if "fetched_at" in cols else ""
                metadata = r[cols.index("metadata")] if has_metadata else "{}"
                if isinstance(metadata, str):
                    try:
                        json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = "{}"
                pg_c.execute(
                    """INSERT INTO request_custom_fields (request_id, client, product, release_version, site, fetched_at, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (request_id) DO NOTHING""",
                    tuple(sanitize(v) for v in (rid, client, product, release_v, site, fetched, metadata)),
                )
            pg.commit()
        count += len(batch)
    print(f"  [OK] request_custom_fields: {count} rows{' (dry run)' if dry_run else ''}")
    return {"custom_fields": count}


def copy_prompts(sqlite, pg, dry_run):
    sq = sqlite.cursor()
    sq.execute("SELECT name, content, is_active, created_at, updated_at FROM prompts ORDER BY name")
    rows = sq.fetchall()
    if not rows:
        print("  - prompts: 0 rows (no data)")
        return {"prompts": 0}
    count = 0
    if not dry_run:
        c = pg.cursor()
        for r in rows:
            c.execute(
                "INSERT INTO prompts (name, content, is_active, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
                tuple(sanitize(v) for v in (r[0], r[1], bool(r[2]), r[3] or datetime.now().isoformat(), r[4] or datetime.now().isoformat())),
            )
        pg.commit()
    count = len(rows)
    print(f"  [OK] prompts: {count} rows{' (dry run)' if dry_run else ''}")
    return {"prompts": count}


def copy_chain_tables(sqlite, pg, dry_run):
    sq = sqlite.cursor()
    sq.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_chains'")
    if not sq.fetchone():
        print("  - prompt_chains: table doesn't exist in SQLite")
        return {"chains": 0, "steps": 0, "runs": 0, "run_steps": 0}

    sq.execute("SELECT id, name, description, created_at, updated_at FROM prompt_chains ORDER BY id")
    chains = sq.fetchall()
    sq.execute("SELECT id, chain_id, step_order, name, prompt_template, input_variable, output_variable, variables FROM prompt_chain_steps ORDER BY id")
    steps = sq.fetchall()
    sq.execute("SELECT id, chain_id, initial_input, status, final_output, error, started_at AS created_at FROM prompt_chain_runs ORDER BY id")
    runs = sq.fetchall()
    sq.execute("SELECT rs.id, rs.run_id, rs.step_order, COALESCE(s.name, ''), rs.input_sent, rs.output_received, rs.status, rs.duration_ms FROM prompt_chain_run_steps rs LEFT JOIN prompt_chain_steps s ON s.id = rs.step_id ORDER BY rs.id")
    run_steps = sq.fetchall()

    if not dry_run:
        c = pg.cursor()
        for r in chains:
            c.execute(
                "INSERT INTO prompt_chains (id, name, description, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                tuple(sanitize(v) for v in r),
            )
        for r in steps:
            vars_json = sanitize(r[7] if isinstance(r[7], str) else json.dumps(r[7] or {}))
            c.execute(
                "INSERT INTO prompt_chain_steps (id, chain_id, step_order, name, prompt_template, input_variable, output_variable, variables) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                tuple(sanitize(v) for v in (r[0], r[1], r[2], r[3], r[4], r[5], r[6], vars_json)),
            )
        for r in runs:
            c.execute(
                "INSERT INTO prompt_chain_runs (id, chain_id, initial_input, status, final_output, error, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                tuple(sanitize(v) for v in r),
            )
        for r in run_steps:
            c.execute(
                "INSERT INTO prompt_chain_run_steps (id, run_id, step_order, name, input_sent, output, status, duration_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                tuple(sanitize(v) for v in r),
            )
        pg.commit()

    print(f"  [OK] prompt_chains: {len(chains)} chains, {len(steps)} steps, {len(runs)} runs, {len(run_steps)} run steps{' (dry run)' if dry_run else ''}")
    return {"chains": len(chains), "steps": len(steps), "runs": len(runs), "run_steps": len(run_steps)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite data to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--sqlite", default=SQLITE_DEFAULT, help="Path to SQLite DB file")
    args = parser.parse_args()

    sqlite_path = os.path.abspath(args.sqlite)
    if not os.path.exists(sqlite_path):
        print(f"ERROR: SQLite database not found: {sqlite_path}")
        sys.exit(1)

    print(f"SQLite source: {sqlite_path}")
    print(f"PostgreSQL target: {PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}")
    print(f"Mode: {'DRY RUN (no writes)' if args.dry_run else 'LIVE'}")
    print()

    # Connect SQLite
    sqlite = sqlite3.connect(sqlite_path)
    sqlite.row_factory = sqlite3.Row

    # Connect PostgreSQL
    try:
        pg = connect_pg()
        print("  [OK] PostgreSQL connection OK")
    except Exception as e:
        print(f"ERROR: Cannot connect to PostgreSQL: {e}")
        print("Make sure PostgreSQL is running and PG_HOST/PG_PORT/PG_DATABASE/PG_USER/PG_PASSWORD env vars are set.")
        sys.exit(1)

    if not args.dry_run:
        create_pg_tables(pg)

    print()
    totals = {}
    totals.update(copy_comments(sqlite, pg, args.dry_run))
    totals.update(copy_summaries(sqlite, pg, args.dry_run))
    totals.update(copy_custom_fields(sqlite, pg, args.dry_run))
    totals.update(copy_prompts(sqlite, pg, args.dry_run))
    totals.update(copy_chain_tables(sqlite, pg, args.dry_run))

    print()
    print("=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print()
    if args.dry_run:
        print("Dry run completed. No data was written to PostgreSQL.")
        print("Re-run without --dry-run to execute.")
    else:
        print("Migration completed successfully.")

    sqlite.close()
    pg.close()


if __name__ == "__main__":
    main()
