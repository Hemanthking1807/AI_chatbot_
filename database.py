"""
database.py — SQLite persistence and SQL Studio layer for Helix AI.

Tables:
  threads        — thread_id, title, created_at, updated_at
  sessions       — created_at, last_active per session_id
  chat_history   — full conversation log per thread and session
  documents      — uploaded document metadata per session
  + Custom user SQL tables created or imported via SQL Studio / CSV import
"""

import sqlite3
import time
import os
import csv
import io
import re

# Database Path Resolution with graceful fallback
_DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "helix_data.db")


def get_db_path() -> str:
    """Returns the active SQLite database path, falling back if path is not writable."""
    env_path = os.getenv("DATABASE_PATH")
    if env_path:
        return env_path

    # Try default path in project folder
    try:
        conn = sqlite3.connect(_DEFAULT_DB, timeout=0.8)
        conn.execute("CREATE TABLE IF NOT EXISTS _write_check (id INTEGER)")
        conn.execute("DROP TABLE IF EXISTS _write_check")
        conn.close()
        return _DEFAULT_DB
    except Exception:
        # In sandboxed / read-only environment, fallback to system temp directory
        import tempfile
        fallback = os.path.join(tempfile.gettempdir(), "helix_data.db")
        return fallback


DATABASE_PATH = get_db_path()


from contextlib import contextmanager

# ─────────────────────────── connection helper ────────────────────────────

def _connect():
    """Return a thread-safe SQLite connection with WAL mode and row factory."""
    global DATABASE_PATH
    DATABASE_PATH = get_db_path()
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    return conn


@contextmanager
def db_session():
    """Context manager that ensures SQLite connections are properly committed and closed."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────── schema init ──────────────────────────────────

def init_db():
    """Create all tables and run automatic migrations if not already present."""
    with db_session() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id   TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   REAL NOT NULL,
                last_active  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL,
                thread_id    TEXT,
                role         TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
                content      TEXT    NOT NULL,
                model_used   TEXT,
                timestamp    REAL    NOT NULL DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS documents (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT    NOT NULL,
                thread_id        TEXT,
                filename         TEXT    NOT NULL,
                file_type        TEXT,
                icon             TEXT,
                pages            INTEGER DEFAULT 1,
                chunks           INTEGER DEFAULT 0,
                file_size        INTEGER DEFAULT 0,
                file_size_fmt    TEXT,
                word_count       INTEGER DEFAULT 0,
                char_count       INTEGER DEFAULT 0,
                upload_time      REAL    NOT NULL DEFAULT (strftime('%s','now')),
                is_active        INTEGER NOT NULL DEFAULT 1
            );
        """)

        # Migration: ensure thread_id column exists on legacy chat_history
        try:
            cursor = conn.execute("PRAGMA table_info(chat_history)")
            cols = [r["name"] for r in cursor.fetchall()]
            if "thread_id" not in cols:
                conn.execute("ALTER TABLE chat_history ADD COLUMN thread_id TEXT")
        except Exception as e:
            print("Migration warning (chat_history):", e)

        # Migration: ensure thread_id column exists on legacy documents
        try:
            cursor = conn.execute("PRAGMA table_info(documents)")
            cols = [r["name"] for r in cursor.fetchall()]
            if "thread_id" not in cols:
                conn.execute("ALTER TABLE documents ADD COLUMN thread_id TEXT")
        except Exception as e:
            print("Migration warning (documents):", e)

        # Create indices after columns are guaranteed to exist
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_chat_session  ON chat_history(session_id);
            CREATE INDEX IF NOT EXISTS idx_chat_thread   ON chat_history(thread_id);
            CREATE INDEX IF NOT EXISTS idx_doc_session   ON documents(session_id);
            CREATE INDEX IF NOT EXISTS idx_threads_time  ON threads(updated_at DESC);
        """)

    print(f"[DB] Database initialised at: {DATABASE_PATH}")


# ─────────────────────────── thread helpers ───────────────────────────────

def create_thread(thread_id: str, title: str = "New Conversation") -> dict:
    """Create a new conversation thread or return existing."""
    now = time.time()
    with db_session() as conn:
        conn.execute("""
            INSERT INTO threads (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
        """, (thread_id, title, now, now))
    return {
        "thread_id": thread_id,
        "title": title,
        "created_at": now,
        "updated_at": now
    }


def get_all_threads() -> list[dict]:
    """Return all conversation threads ordered by most recently updated."""
    with db_session() as conn:
        rows = conn.execute("""
            SELECT t.thread_id, t.title, t.created_at, t.updated_at,
                   COUNT(c.id) AS message_count
            FROM threads t
            LEFT JOIN chat_history c ON t.thread_id = c.thread_id
            GROUP BY t.thread_id
            ORDER BY t.updated_at DESC
        """).fetchall()
    threads = []
    for r in rows:
        d = dict(r)
        d["created_at_fmt"] = _fmt_datetime(d["created_at"])
        d["updated_at_fmt"] = _fmt_datetime(d["updated_at"])
        threads.append(d)
    return threads


def get_thread(thread_id: str) -> dict | None:
    """Return single thread record by thread_id."""
    with db_session() as conn:
        row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
    if row:
        d = dict(row)
        d["created_at_fmt"] = _fmt_datetime(d["created_at"])
        d["updated_at_fmt"] = _fmt_datetime(d["updated_at"])
        return d
    return None


def update_thread_title(thread_id: str, title: str):
    """Rename a thread's title."""
    clean_title = (title or "Untitled Conversation").strip()[:80]
    with db_session() as conn:
        conn.execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
            (clean_title, time.time(), thread_id)
        )


def touch_thread(thread_id: str):
    """Update updated_at timestamp for a thread."""
    with db_session() as conn:
        conn.execute(
            "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
            (time.time(), thread_id)
        )


def delete_thread(thread_id: str):
    """Permanently delete a thread and all of its associated messages."""
    with db_session() as conn:
        conn.execute("DELETE FROM chat_history WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))


def generate_title_from_text(text: str) -> str:
    """Generate a clean, readable thread title from the initial user prompt."""
    if not text:
        return "New Conversation"
    # Strip markdown headers or code block ticks
    clean = re.sub(r'^[#*`\s]+', '', text).strip()
    clean = clean.split("\n")[0].strip()
    if len(clean) > 42:
        clean = clean[:42].rsplit(" ", 1)[0] + "..."
    return clean or "New Conversation"


# ─────────────────────────── session helpers ──────────────────────────────

def ensure_session(session_id: str):
    """Insert a new session row or update last_active timestamp."""
    now = time.time()
    with db_session() as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, created_at, last_active)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
        """, (session_id, now, now))


def touch_session(session_id: str):
    """Update the last_active timestamp for an existing session."""
    with db_session() as conn:
        conn.execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (time.time(), session_id)
        )


def get_all_sessions():
    """Return all sessions ordered by most recently active."""
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY last_active DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────── chat history ─────────────────────────────────

def save_message(session_id: str, role: str, content: str, model_used: str = None, thread_id: str = None):
    """
    Persist a single chat turn (user or assistant) associated with session and thread.
    """
    ensure_session(session_id)
    if thread_id:
        ensure_thread_exists(thread_id, title=generate_title_from_text(content) if role == "user" else None)
        touch_thread(thread_id)

    with db_session() as conn:
        conn.execute("""
            INSERT INTO chat_history (session_id, thread_id, role, content, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, thread_id, role, content, model_used, time.time()))


def ensure_thread_exists(thread_id: str, title: str = None):
    """Ensure thread record exists without overwriting existing title."""
    now = time.time()
    with db_session() as conn:
        row = conn.execute("SELECT thread_id, title FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if not row:
            conn.execute("""
                INSERT INTO threads (thread_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (thread_id, title or "New Conversation", now, now))
        elif title and row["title"] == "New Conversation":
            conn.execute("UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?", (title, now, thread_id))


def get_history(session_id: str = None, thread_id: str = None, limit: int = 100) -> list[dict]:
    """Retrieve conversation history for a session or thread, oldest first."""
    with db_session() as conn:
        if thread_id:
            rows = conn.execute("""
                SELECT id, session_id, thread_id, role, content, model_used, timestamp
                FROM chat_history
                WHERE thread_id = ?
                ORDER BY id ASC
                LIMIT ?
            """, (thread_id, limit)).fetchall()
        elif session_id:
            rows = conn.execute("""
                SELECT id, session_id, thread_id, role, content, model_used, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
            """, (session_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, session_id, thread_id, role, content, model_used, timestamp
                FROM chat_history
                ORDER BY id ASC
                LIMIT ?
            """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_history_as_turns(session_id: str = None, thread_id: str = None, limit: int = 50) -> list[dict]:
    """
    Return history in paired {user, assistant, timestamp, model} turns.
    """
    rows = get_history(session_id=session_id, thread_id=thread_id, limit=limit * 2)
    turns = []
    i = 0
    while i < len(rows):
        user_row = rows[i] if rows[i]["role"] == "user" else None
        asst_row = rows[i + 1] if i + 1 < len(rows) and rows[i + 1]["role"] == "assistant" else None

        if user_row and asst_row:
            turns.append({
                "user":      user_row["content"],
                "assistant": asst_row["content"],
                "timestamp": _fmt_time(user_row["timestamp"]),
                "model":     asst_row.get("model_used", ""),
                "thread_id": user_row.get("thread_id"),
            })
            i += 2
        else:
            i += 1

    return turns[-limit:]


def clear_session_history(session_id: str, thread_id: str = None):
    """Delete messages for session or specific thread."""
    with db_session() as conn:
        if thread_id:
            conn.execute("DELETE FROM chat_history WHERE thread_id = ?", (thread_id,))
            conn.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (time.time(), thread_id))
        elif session_id:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))


# ─────────────────────────── document metadata ────────────────────────────

def save_document(session_id: str, metadata: dict, thread_id: str = None):
    """Persist document metadata and mark all previous docs as inactive for this session."""
    ensure_session(session_id)
    with db_session() as conn:
        conn.execute("UPDATE documents SET is_active = 0 WHERE session_id = ?", (session_id,))
        conn.execute("""
            INSERT INTO documents
                (session_id, thread_id, filename, file_type, icon, pages, chunks,
                 file_size, file_size_fmt, word_count, char_count, upload_time, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            session_id,
            thread_id,
            metadata.get("filename", ""),
            metadata.get("file_type", ""),
            metadata.get("icon", "📄"),
            metadata.get("pages", 1),
            metadata.get("chunks", 0),
            metadata.get("file_size", 0),
            metadata.get("file_size_formatted", ""),
            metadata.get("word_count", 0),
            metadata.get("char_count", 0),
            time.time(),
        ))


def get_document(session_id: str) -> dict | None:
    """Return the most recently active document for a session, or None."""
    with db_session() as conn:
        row = conn.execute("""
            SELECT * FROM documents
            WHERE session_id = ? AND is_active = 1
            ORDER BY upload_time DESC
            LIMIT 1
        """, (session_id,)).fetchone()
    return dict(row) if row else None


def deactivate_document(session_id: str):
    """Mark the active document for a session as removed."""
    with db_session() as conn:
        conn.execute("UPDATE documents SET is_active = 0 WHERE session_id = ?", (session_id,))


def get_all_documents(session_id: str = None) -> list[dict]:
    """Return upload history (newest first)."""
    with db_session() as conn:
        if session_id:
            rows = conn.execute("""
                SELECT * FROM documents WHERE session_id = ? ORDER BY upload_time DESC
            """, (session_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents ORDER BY upload_time DESC").fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────── SQL Studio Engine ────────────────────────────

def execute_user_query(sql_query: str, max_rows: int = 200) -> dict:
    """
    Execute user SQL query with safe limits and full introspection.
    Supports SELECT, PRAGMA, EXPLAIN, and table creation/queries.
    """
    clean_query = (sql_query or "").strip()
    if not clean_query:
        return {"success": False, "error": "Query cannot be empty."}

    # Remove trailing semicolons for uniform processing
    clean_query = clean_query.rstrip(";")

    start_time = time.time()
    try:
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute(clean_query)

            # Check if query returned rows (e.g. SELECT, PRAGMA)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                raw_rows = cursor.fetchmany(max_rows)
                rows = [list(r) for r in raw_rows]
                elapsed = time.time() - start_time
                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "execution_time_ms": round(elapsed * 1000, 2),
                    "query": clean_query,
                    "is_select": True,
                }
            else:
                conn.commit()
                elapsed = time.time() - start_time
                affected = cursor.rowcount
                return {
                    "success": True,
                    "columns": ["Result"],
                    "rows": [[f"Query executed successfully. Affected rows: {affected}"]],
                    "row_count": max(0, affected),
                    "execution_time_ms": round(elapsed * 1000, 2),
                    "query": clean_query,
                    "is_select": False,
                }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "execution_time_ms": round(elapsed * 1000, 2),
            "query": clean_query,
        }


def get_all_tables_info() -> list[dict]:
    """
    Return comprehensive metadata for all tables in SQLite:
    Name, row count, column definitions, and primary keys.
    """
    tables = []
    with db_session() as conn:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name ASC
        """)
        table_names = [r["name"] for r in cursor.fetchall()]

        for name in table_names:
            # Get row count
            try:
                cnt = conn.execute(f"SELECT COUNT(*) FROM `{name}`").fetchone()[0]
            except Exception:
                cnt = 0

            # Get column schema
            cols_cursor = conn.execute(f"PRAGMA table_info(`{name}`)")
            columns = []
            for col in cols_cursor.fetchall():
                columns.append({
                    "cid": col["cid"],
                    "name": col["name"],
                    "type": col["type"] or "TEXT",
                    "notnull": bool(col["notnull"]),
                    "dflt_value": col["dflt_value"],
                    "pk": bool(col["pk"]),
                })

            tables.append({
                "table_name": name,
                "row_count": cnt,
                "columns": columns,
            })

    return tables


def import_csv_to_table(table_name: str, csv_data: str) -> dict:
    """
    Create a new SQL table from CSV text data and populate rows.
    Automatically cleans column names and infers data types.
    """
    # Clean table name to alphanumeric/underscore
    clean_table = re.sub(r"[^a-zA-Z0-9_]", "_", table_name.strip()).strip("_")
    if not clean_table:
        clean_table = f"imported_data_{int(time.time())}"

    f = io.StringIO(csv_data.strip())
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        return {"success": False, "error": "CSV file is empty."}

    clean_columns = []
    seen = set()
    for i, col in enumerate(header):
        c = re.sub(r"[^a-zA-Z0-9_]", "_", col.strip()).strip("_")
        if not c:
            c = f"column_{i+1}"
        orig = c
        idx = 1
        while c in seen:
            c = f"{orig}_{idx}"
            idx += 1
        seen.add(c)
        clean_columns.append(c)

    rows = []
    for r in reader:
        if r and any(cell.strip() for cell in r):
            # Pad or trim row to match column count
            padded = r[:len(clean_columns)] + [""] * max(0, len(clean_columns) - len(r))
            rows.append(padded)

    with db_session() as conn:
        # Drop if already exists, recreate
        cols_def = ", ".join([f"`{c}` TEXT" for c in clean_columns])
        conn.execute(f"DROP TABLE IF EXISTS `{clean_table}`")
        conn.execute(f"CREATE TABLE `{clean_table}` ({cols_def})")

        placeholders = ", ".join(["?"] * len(clean_columns))
        conn.executemany(
            f"INSERT INTO `{clean_table}` VALUES ({placeholders})",
            rows
        )
        conn.commit()

    return {
        "success": True,
        "table_name": clean_table,
        "columns": clean_columns,
        "row_count": len(rows),
    }


def get_stats() -> dict:
    """Return database metrics and analytics."""
    with db_session() as conn:
        total_threads   = conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
        total_sessions  = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_messages  = conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
        total_documents = conn.execute("SELECT COUNT(*) FROM documents WHERE is_active = 1").fetchone()[0]
        total_tables    = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone()[0]

    db_size = 0
    try:
        if os.path.exists(DATABASE_PATH):
            db_size = os.path.getsize(DATABASE_PATH)
    except Exception:
        pass

    size_fmt = f"{db_size / 1024:.1f} KB" if db_size < 1024 * 1024 else f"{db_size / (1024 * 1024):.2f} MB"

    return {
        "total_threads":         total_threads,
        "total_sessions":        total_sessions,
        "total_messages":        total_messages,
        "active_documents":      total_documents,
        "total_tables":          total_tables,
        "db_size_bytes":         db_size,
        "db_size_formatted":     size_fmt,
        "db_path":               DATABASE_PATH,
        "sqlite_version":        sqlite3.sqlite_version,
    }


# ─────────────────────────── utilities ────────────────────────────────────

def _fmt_time(ts: float) -> str:
    """Convert a Unix timestamp to HH:MM string."""
    import datetime
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    except Exception:
        return ""


def _fmt_datetime(ts: float) -> str:
    """Convert a Unix timestamp to readable date/time string."""
    import datetime
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%b %d, %H:%M")
    except Exception:
        return ""
