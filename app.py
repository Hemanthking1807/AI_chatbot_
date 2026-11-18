import io
import os
import uuid
import time
import re
from flask import Flask, request, jsonify, render_template, session, Response, send_file
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import database as db

# Load environment variables
load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    app.secret_key = os.urandom(32).hex()

# Initialise SQLite database (creates tables & runs migrations if not exist)
db.init_db()

# Gemini API setup
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = None
if api_key and not (api_key.startswith("ya29.") or api_key.startswith("1//")):
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("Warning: Gemini client initialization error:", repr(e))
else:
    print("Warning: GEMINI_API_KEY is not set or has an invalid format.")

# Preferred model list with automatic fallback
PREFERRED_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-pro-latest",
]
MODEL_FALLBACK_LIST = list(dict.fromkeys([m for m in PREFERRED_MODELS if m]))

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB limit

# Allowed extensions: PDF, text, documentation, code, and database files
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts",
    ".html", ".css", ".xml", ".log", ".yaml", ".yml", ".tsv",
    ".sql", ".sh", ".bat", ".env", ".java", ".c", ".cpp", ".cs",
    ".php", ".rb", ".go", ".rs", ".rtf", ".text", ".sqlite", ".db"
}

# In-memory session document store
documents = {}
MAX_STORED_DOCUMENTS = 60
MAX_HISTORY_TURNS = 12
MAX_FULL_TEXT_DIRECT_CHARS = 35000  # Pass full text directly if under ~35,000 characters
MAX_RELEVANT_CHUNKS = 5
MAX_CONTEXT_CHARS = 24000
MAX_OUTPUT_TOKENS = 1500


def cleanup_old_documents():
    """Prunes stale document indexes to prevent memory leaks."""
    now = time.time()
    stale_keys = [
        sid for sid, data in documents.items()
        if now - data.get("timestamp", now) > 14400
    ]
    for sid in stale_keys:
        documents.pop(sid, None)

    if len(documents) > MAX_STORED_DOCUMENTS:
        sorted_keys = sorted(
            documents.keys(),
            key=lambda k: documents[k].get("timestamp", 0),
            reverse=True,
        )
        for old_key in sorted_keys[MAX_STORED_DOCUMENTS:]:
            documents.pop(old_key, None)


def format_gemini_error(error):
    """Generates user-friendly diagnostics for Gemini API exceptions."""
    error_str = str(error)
    error_text = error_str.lower()

    if "unauthenticated" in error_text or "api key not valid" in error_text or "invalid api key" in error_text:
        return (
            "The Gemini API key appears invalid or expired. "
            "Please check your GEMINI_API_KEY in the .env file."
        )

    if (
        "429" in error_str
        or "resource_exhausted" in error_text
        or "quota exceeded" in error_text
        or "exceeded your current quota" in error_text
    ):
        return (
            "Google Gemini API quota has been reached. "
            "Please wait a moment before asking another question, or check your API plan."
        )

    if "connecterror" in error_text or "11001" in error_text or "network" in error_text or "getaddrinfo" in error_text:
        return (
            "Network connection failed. Please ensure your device has internet access "
            "and can reach Google Gemini services."
        )

    if "503" in error_str or "unavailable" in error_text or "high demand" in error_text:
        return (
            "Google AI services are experiencing temporary high demand. "
            "Retrying with alternative model..."
        )

    return f"AI service error: {error_str}"


def split_text(text, chunk_size=400, overlap=80):
    """Splits raw text into overlapping word chunks for TF-IDF retrieval."""
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def build_document_index(chunks, full_text="", metadata=None):
    """Builds a TF-IDF vectorizer and matrix for semantic search over chunks."""
    if not chunks:
        return {
            "chunks": [],
            "full_text": full_text or "",
            "vectorizer": None,
            "tfidf_matrix": None,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)
    except Exception:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(chunks)

    return {
        "chunks": chunks,
        "full_text": full_text or "",
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "metadata": metadata or {},
        "timestamp": time.time(),
    }


def retrieve_chunks(question, doc_entry, top_k=5, min_similarity=0.01):
    """Finds top-k most relevant document chunks for a given query."""
    if not question or not doc_entry:
        return []

    if isinstance(doc_entry, dict):
        chunk_texts = doc_entry.get("chunks", [])
        vectorizer = doc_entry.get("vectorizer")
        tfidf_matrix = doc_entry.get("tfidf_matrix")
        if not chunk_texts or vectorizer is None or tfidf_matrix is None:
            return chunk_texts[:top_k]
    else:
        chunk_texts = doc_entry
        if not chunk_texts:
            return []
        index_data = build_document_index(chunk_texts)
        vectorizer = index_data["vectorizer"]
        tfidf_matrix = index_data["tfidf_matrix"]

    try:
        query_vector = vectorizer.transform([question])
        similarity = cosine_similarity(query_vector, tfidf_matrix)[0]

        ranked_indices = similarity.argsort()[::-1]
        relevant = [
            chunk_texts[i]
            for i in ranked_indices[:top_k]
            if similarity[i] >= min_similarity
        ]

        # If question is general (e.g. summary, overview), fallback to first chunks
        if not relevant and chunk_texts:
            return chunk_texts[:top_k]

        return relevant
    except Exception:
        return chunk_texts[:top_k] if chunk_texts else []


def extract_file_content(file_bytes, filename):
    """Extracts text content and metadata from PDF, text/code, or database/CSV files."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            stream = io.BytesIO(file_bytes)
            reader = PdfReader(stream, strict=False)
            pages_count = len(reader.pages)
            if pages_count == 0:
                raise ValueError("The PDF file does not contain any pages.")

            extracted_pages = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    clean_page = text.strip()
                    if clean_page:
                        extracted_pages.append(f"[Page {i + 1}]\n{clean_page}")
                except Exception as pe:
                    print(f"Warning: page {i+1} extraction failed: {pe}")

            full_text = "\n\n".join(extracted_pages).strip()
            if not full_text:
                raise ValueError(
                    "No readable text found in this PDF. It might be scanned or image-only."
                )

            return {
                "text": full_text,
                "pages": pages_count,
                "file_type": "PDF Document",
                "icon": "📄",
            }
        except Exception as e:
            if "scanned" in str(e).lower() or "no readable text" in str(e).lower():
                raise e
            raise ValueError(f"Failed to read PDF: {str(e)}")

    else:
        # Text, Markdown, CSV, JSON, Code, Log, SQL files
        full_text = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                full_text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if full_text is None:
            full_text = file_bytes.decode("utf-8", errors="replace")

        full_text = full_text.strip()
        if not full_text:
            raise ValueError("The uploaded text file is empty.")

        lines = len(full_text.splitlines())
        pages = max(1, (lines + 40) // 45)

        type_map = {
            ".txt": ("Text Document", "📝"),
            ".md": ("Markdown Document", "📝"),
            ".csv": ("CSV Dataset", "📊"),
            ".json": ("JSON File", "💻"),
            ".py": ("Python Code", "🐍"),
            ".js": ("JavaScript Code", "💻"),
            ".ts": ("TypeScript Code", "💻"),
            ".html": ("HTML File", "🌐"),
            ".css": ("CSS Stylesheet", "🎨"),
            ".sql": ("SQL Script", "🗄️"),
            ".log": ("Log File", "📋"),
            ".sqlite": ("SQLite Database", "🗄️"),
            ".db": ("SQLite Database", "🗄️"),
        }
        file_type, icon = type_map.get(ext, ("Text Document", "📄"))

        return {
            "text": full_text,
            "pages": pages,
            "file_type": file_type,
            "icon": icon,
        }


def generate_with_fallback(prompt, system_instruction=None, preferred_model=None):
    """Attempts generation with primary model, then falls back to secondary models if needed."""
    global client
    if client is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured in the environment or .env file.")
        client = genai.Client(api_key=key)

    models_to_try = MODEL_FALLBACK_LIST
    if preferred_model:
        models_to_try = [preferred_model] + [m for m in MODEL_FALLBACK_LIST if m != preferred_model]

    last_error = None
    for model_name in models_to_try:
        try:
            config = types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            text_output = getattr(response, "text", None)
            if text_output is None:
                try:
                    text_output = response.candidates[0].content.parts[0].text
                except (AttributeError, IndexError, TypeError):
                    text_output = ""

            if text_output:
                return text_output, model_name
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            print(f"Model {model_name} failed: {err_str[:120]}")
            if any(k in err_str for k in ["404", "not_found", "503", "unavailable", "high demand", "overloaded"]):
                continue
            if "unauthenticated" in err_str or "api key not valid" in err_str:
                raise e
            continue

    if last_error:
        raise last_error
    raise RuntimeError("No models responded.")


def format_sql_markdown_table(result: dict) -> str:
    """Formats an execute_user_query result dictionary into a GitHub Markdown table."""
    if not result.get("success"):
        return f"⚠️ **SQL Error:**\n```\n{result.get('error', 'Unknown query execution error')}\n```"

    cols = result.get("columns", [])
    rows = result.get("rows", [])
    time_ms = result.get("execution_time_ms", 0)
    row_cnt = result.get("row_count", 0)
    query = result.get("query", "")

    header = f"🗄️ **SQL Query Executed:**\n```sql\n{query}\n```\n*(⏱️ {time_ms} ms • 📊 {row_cnt} row{'s' if row_cnt != 1 else ''})*\n\n"

    if not cols:
        return header + "*Query executed successfully with no returned rows.*"

    # Markdown table header
    table_lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join([":---"] * len(cols)) + " |"
    ]

    for row in rows[:50]:  # Cap at 50 for chat display
        row_cells = []
        for cell in row:
            val_str = str(cell) if cell is not None else "*NULL*"
            # Escape pipes and newlines for markdown table
            clean_cell = val_str.replace("|", "\\|").replace("\n", " ").strip()
            if len(clean_cell) > 120:
                clean_cell = clean_cell[:117] + "..."
            row_cells.append(clean_cell)
        table_lines.append("| " + " | ".join(row_cells) + " |")

    if len(rows) > 50:
        table_lines.append(f"\n*(Displaying first 50 of {row_cnt} total rows. View full table in SQL Studio)*")

    return header + "\n".join(table_lines)


# ---------------- Routes ----------------


@app.route("/")
def home():
    """Renders the single-page chat interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health and diagnostic endpoint."""
    db_stats = {}
    try:
        db_stats = db.get_stats()
    except Exception as e:
        db_stats = {"error": str(e)}

    return jsonify({
        "status": "online",
        "has_api_key": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "models": MODEL_FALLBACK_LIST,
        "active_sessions": len(documents),
        "database": db_stats,
    })


# ─────────────────────────── Thread Management ────────────────────────────

@app.route("/threads", methods=["GET"])
def get_threads():
    """Returns all conversation threads stored in the SQL database."""
    try:
        threads = db.get_all_threads()
        current_thread_id = session.get("thread_id")
        return jsonify({
            "threads": threads,
            "current_thread_id": current_thread_id
        })
    except Exception as e:
        return jsonify({"error": str(e), "threads": []}), 500


@app.route("/threads", methods=["POST"])
def create_new_thread():
    """Creates a new conversation thread in SQLite and activates it."""
    try:
        data = request.get_json(silent=True) or {}
        title = data.get("title") or "New Conversation"
        new_thread_id = str(uuid.uuid4())
        thread_record = db.create_thread(new_thread_id, title)
        session["thread_id"] = new_thread_id
        session.pop("history", None)
        return jsonify({
            "success": True,
            "thread": thread_record
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/threads/switch", methods=["POST"])
def switch_thread():
    """Switches the active conversation thread."""
    try:
        data = request.get_json(silent=True) or {}
        target_id = data.get("thread_id")
        if not target_id:
            return jsonify({"error": "thread_id is required"}), 400

        thread = db.get_thread(target_id)
        if not thread:
            # If not in DB, create it
            thread = db.create_thread(target_id, "Conversation")

        session["thread_id"] = target_id
        history = db.get_history_as_turns(thread_id=target_id, limit=MAX_HISTORY_TURNS * 4)

        return jsonify({
            "success": True,
            "thread_id": target_id,
            "thread": thread,
            "history": history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/threads/<thread_id>", methods=["PATCH"])
def rename_thread(thread_id):
    """Renames an existing conversation thread."""
    try:
        data = request.get_json(silent=True) or {}
        title = data.get("title", "").strip()
        if not title:
            return jsonify({"error": "Title cannot be empty"}), 400

        db.update_thread_title(thread_id, title)
        return jsonify({"success": True, "thread_id": thread_id, "title": title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/threads/<thread_id>", methods=["DELETE"])
def delete_thread(thread_id):
    """Deletes a conversation thread and its messages from SQLite."""
    try:
        db.delete_thread(thread_id)
        # If current thread was deleted, reset thread_id in session
        if session.get("thread_id") == thread_id:
            session.pop("thread_id", None)
            session.pop("history", None)
        return jsonify({"success": True, "message": "Thread deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── Document Routes ──────────────────────────────

@app.route("/document", methods=["GET"])
def get_document_info():
    """Returns metadata about the active document in the current session."""
    session_id = session.get("session_id")
    if not session_id or session_id not in documents:
        return jsonify({"has_document": False})

    doc_data = documents[session_id]
    metadata = doc_data.get("metadata", {})
    return jsonify({
        "has_document": True,
        "filename": metadata.get("filename", "Uploaded File"),
        "pages": metadata.get("pages", 1),
        "chunks": len(doc_data.get("chunks", [])),
        "file_size_formatted": metadata.get("file_size_formatted", ""),
        "file_type": metadata.get("file_type", "Document"),
        "icon": metadata.get("icon", "📄"),
        "word_count": metadata.get("word_count", 0),
        "char_count": metadata.get("char_count", 0),
    })


@app.route("/document/preview", methods=["GET"])
def preview_document():
    """Returns extracted text preview and chunk structure for the active document."""
    session_id = session.get("session_id")
    if not session_id or session_id not in documents:
        return jsonify({"error": "No document is currently attached."}), 404

    doc_data = documents[session_id]
    metadata = doc_data.get("metadata", {})
    chunks = doc_data.get("chunks", [])
    full_text = doc_data.get("full_text", "")

    return jsonify({
        "filename": metadata.get("filename", "Document"),
        "file_type": metadata.get("file_type", "Document"),
        "file_size": metadata.get("file_size_formatted", ""),
        "pages": metadata.get("pages", 1),
        "total_chunks": len(chunks),
        "word_count": metadata.get("word_count", len(full_text.split())),
        "char_count": len(full_text),
        "preview_text": full_text[:1400] + ("..." if len(full_text) > 1400 else ""),
        "sample_chunks": [c[:280] + "..." for c in chunks[:3]],
    })


@app.route("/document", methods=["DELETE"])
def remove_document():
    """Detaches/removes the uploaded document from the current session."""
    session_id = session.get("session_id")
    if session_id and session_id in documents:
        documents.pop(session_id, None)
    if session_id:
        try:
            db.deactivate_document(session_id)
        except Exception as db_err:
            print("DB deactivate_document warning:", repr(db_err))
    return jsonify({"message": "Document removed successfully"})


@app.route("/upload", methods=["POST"])
def upload_file():
    """Extracts text from an uploaded file, indexes it for RAG, and optionally imports CSV/SQL into SQLite."""
    try:
        cleanup_old_documents()

        if "file" not in request.files:
            return jsonify({"error": "No file attached to the request"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Please select a file to upload"}), 400

        original_filename = file.filename
        ext = os.path.splitext(original_filename)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
            return jsonify({
                "error": f"Unsupported file type '{ext}'. Supported formats: PDF, TXT, MD, CSV, JSON, Code, SQL files."
            }), 400

        file_bytes = file.read()
        file_size = len(file_bytes)
        if file_size == 0:
            return jsonify({"error": "The uploaded file is empty (0 bytes)."}), 400

        # Extract file text & metadata
        try:
            parsed = extract_file_content(file_bytes, original_filename)
            full_text = parsed["text"]
            pages_count = parsed["pages"]
            file_type = parsed["file_type"]
            icon = parsed["icon"]
        except Exception as e:
            print("File Extraction Error:", repr(e))
            return jsonify({"error": str(e)}), 400

        chunks = split_text(full_text, chunk_size=400, overlap=80)
        if not chunks:
            return jsonify({"error": "Unable to extract readable text chunks from this file."}), 400

        # Session & Thread setup
        session_id = session.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            session["session_id"] = session_id

        thread_id = session.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            session["thread_id"] = thread_id
            db.create_thread(thread_id, f"File: {original_filename}")

        if file_size < 1024 * 1024:
            size_fmt = f"{file_size / 1024:.1f} KB"
        else:
            size_fmt = f"{file_size / (1024 * 1024):.1f} MB"

        metadata = {
            "filename": original_filename,
            "pages": pages_count,
            "file_size": file_size,
            "file_size_formatted": size_fmt,
            "file_type": file_type,
            "icon": icon,
            "char_count": len(full_text),
            "word_count": len(full_text.split()),
        }

        # Store in memory for RAG
        documents[session_id] = build_document_index(
            chunks,
            full_text=full_text,
            metadata=metadata
        )

        # Persist document metadata to database
        try:
            db.save_document(session_id, {**metadata, "chunks": len(chunks)}, thread_id=thread_id)
        except Exception as db_err:
            print("DB save_document warning:", repr(db_err))

        # If it's a CSV file, also automatically import it as an SQL table for immediate querying!
        csv_sql_table = None
        if ext == ".csv":
            try:
                base_name = os.path.splitext(original_filename)[0]
                csv_result = db.import_csv_to_table(base_name, full_text)
                if csv_result.get("success"):
                    csv_sql_table = csv_result.get("table_name")
            except Exception as e:
                print("CSV SQL import warning:", repr(e))

        resp_data = {
            "message": f"{file_type} uploaded and indexed successfully!",
            "filename": original_filename,
            "pages": pages_count,
            "chunks": len(chunks),
            "file_size_formatted": size_fmt,
            "file_type": file_type,
            "icon": icon,
            "word_count": metadata["word_count"],
        }
        if csv_sql_table:
            resp_data["sql_table"] = csv_sql_table
            resp_data["message"] += f" Also imported into SQL database as table `{csv_sql_table}`!"

        return jsonify(resp_data)

    except Exception as e:
        print("Upload Handler Error:", repr(e))
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


# ─────────────────────────── Chat Flow & SQL Commands ─────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """Processes user queries with document context, conversation history, and SQL execution."""
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "")

        if not isinstance(user_message, str):
            return jsonify({"error": "Invalid message format."}), 400

        user_message = user_message.strip()
        if not user_message:
            return jsonify({"error": "Please enter a message or question."}), 400

        # Session & Thread management
        session_id = session.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            session["session_id"] = session_id

        thread_id = session.get("thread_id")
        is_first_thread_turn = False
        if not thread_id:
            thread_id = str(uuid.uuid4())
            session["thread_id"] = thread_id
            title = db.generate_title_from_text(user_message)
            db.create_thread(thread_id, title)
            is_first_thread_turn = True
        else:
            # Check if thread is currently "New Conversation" to update its title
            existing_thread = db.get_thread(thread_id)
            if existing_thread and existing_thread.get("title") == "New Conversation":
                db.update_thread_title(thread_id, db.generate_title_from_text(user_message))

        # Check for direct SQL command: /sql <query>
        if user_message.lower().startswith("/sql ") or user_message.lower() == "/sql":
            raw_sql = user_message[4:].strip()
            if not raw_sql:
                return jsonify({
                    "response": "Please enter an SQL query after `/sql`. Example: `/sql SELECT * FROM threads LIMIT 5;`",
                    "model": "SQL Engine",
                    "is_sql": True,
                })

            result = db.execute_user_query(raw_sql)
            formatted_output = format_sql_markdown_table(result)

            # Persist to database
            try:
                db.save_message(session_id, "user", user_message, thread_id=thread_id)
                db.save_message(session_id, "assistant", formatted_output, model_used="SQLite Engine", thread_id=thread_id)
                db.touch_session(session_id)
            except Exception as db_err:
                print("DB save_message warning:", repr(db_err))

            return jsonify({
                "response": formatted_output,
                "model": "SQLite Engine",
                "used_document": False,
                "is_sql": True,
                "thread_id": thread_id,
            })

        # Retrieve recent conversation history from DB for this thread
        db_history = []
        try:
            db_history = db.get_history_as_turns(thread_id=thread_id, limit=MAX_HISTORY_TURNS)
        except Exception:
            pass

        history = db_history if db_history else session.get("history", [])
        history = history[-MAX_HISTORY_TURNS:]

        # Check for active document
        doc_entry = documents.get(session_id)
        doc_name = None
        doc_meta = {}
        used_document = False
        context_str = ""

        if doc_entry and (doc_entry.get("chunks") or doc_entry.get("full_text")):
            used_document = True
            doc_meta = doc_entry.get("metadata", {})
            doc_name = doc_meta.get("filename", "Uploaded File")
            full_text = doc_entry.get("full_text", "")

            if len(full_text) <= MAX_FULL_TEXT_DIRECT_CHARS:
                context_str = full_text
            else:
                relevant_chunks = retrieve_chunks(
                    user_message,
                    doc_entry,
                    top_k=MAX_RELEVANT_CHUNKS,
                    min_similarity=0.01,
                )
                if not relevant_chunks:
                    relevant_chunks = doc_entry.get("chunks", [])[:4]

                intro = full_text[:2000]
                context_str = (
                    f"[DOCUMENT BEGINNING EXCERPT]\n{intro}\n\n"
                    f"[RELEVANT DOCUMENT SECTIONS]\n" + "\n\n---\n\n".join(relevant_chunks)
                )

            if len(context_str) > MAX_CONTEXT_CHARS:
                context_str = context_str[:MAX_CONTEXT_CHARS]

        # Gather database tables overview to empower Text-to-SQL
        db_schema_snippet = ""
        try:
            tables_info = db.get_all_tables_info()
            if tables_info:
                table_summaries = []
                for t in tables_info[:8]:
                    cols_desc = ", ".join([f"{c['name']} ({c['type']})" for c in t['columns'][:8]])
                    table_summaries.append(f"- `{t['table_name']}` ({t['row_count']} rows): {cols_desc}")
                db_schema_snippet = "### APPLICATION SQL DATABASE (SQLite):\n" + "\n".join(table_summaries)
        except Exception:
            pass

        # Construct prompt & system instruction
        base_instructions = (
            "You are Helix AI, a next-generation articulate, knowledgeable, and highly capable AI assistant. "
            "You have integrated Document QA (RAG) and a live SQLite Database Studio. "
            "Format your responses cleanly using GitHub-Flavored Markdown: clear headings, bullet points, and code blocks with syntax highlighting where relevant. "
            "If the user asks about database data, queries, or SQL, provide accurate SQL queries or insights. Users can also run queries directly with `/sql <query>` or via the SQL Studio modal.\n\n"
            f"{db_schema_snippet}\n"
        )

        if used_document:
            system_instruction = (
                base_instructions +
                f"\nThe user has provided an uploaded file titled '{doc_name}' ({doc_meta.get('file_type', 'Document')}). "
                "Base your answers primarily on the DOCUMENT CONTENT provided below. Provide structured, comprehensive, and clear answers."
            )
            prompt = (
                f"### ATTACHED FILE: {doc_name} ({doc_meta.get('file_type', 'Document')})\n\n"
                f"### DOCUMENT CONTENT:\n{context_str}\n\n"
                f"### RECENT CONVERSATION HISTORY:\n{history}\n\n"
                f"### USER INQUIRY:\n{user_message}"
            )
        else:
            system_instruction = base_instructions
            prompt = (
                f"### RECENT CONVERSATION HISTORY:\n{history}\n\n"
                f"### USER INQUIRY:\n{user_message}"
            )

        requested_model = data.get("model")

        # Generate response using fallback model chain
        ai_response, model_used = generate_with_fallback(
            prompt,
            system_instruction=system_instruction,
            preferred_model=requested_model,
        )

        # Persist messages to SQLite database
        try:
            db.save_message(session_id, "user", user_message, thread_id=thread_id)
            db.save_message(session_id, "assistant", ai_response, model_used=model_used, thread_id=thread_id)
            db.touch_session(session_id)
        except Exception as db_err:
            print("DB save_message warning:", repr(db_err))

        # Keep slim in-session copy
        slim_turn = {
            "user": user_message[:200],
            "assistant": ai_response[:200],
            "timestamp": time.strftime("%H:%M"),
        }
        slim_history = session.get("history", [])
        slim_history.append(slim_turn)
        session["history"] = slim_history[-4:]

        return jsonify({
            "response": ai_response,
            "model": model_used,
            "used_document": used_document,
            "document_name": doc_name,
            "thread_id": thread_id,
        })

    except Exception as e:
        print("Chat Error:", repr(e))
        error_msg = format_gemini_error(e)
        status_code = 429 if "quota" in error_msg.lower() else (
            401 if "api key" in error_msg.lower() else 500
        )
        return jsonify({"error": error_msg}), status_code


# ─────────────────────────── History & Export ─────────────────────────────

@app.route("/history", methods=["GET"])
def get_history_route():
    """Returns the conversation history for the current thread or session from SQLite."""
    thread_id = request.args.get("thread_id") or session.get("thread_id")
    session_id = session.get("session_id")

    history = []
    if thread_id:
        try:
            history = db.get_history_as_turns(thread_id=thread_id, limit=MAX_HISTORY_TURNS * 4)
        except Exception:
            pass

    if not history and session_id:
        try:
            history = db.get_history_as_turns(session_id=session_id, limit=MAX_HISTORY_TURNS * 4)
        except Exception:
            pass

    # Cookie fallback if database returned empty
    if not history:
        history = session.get("history", [])

    return jsonify({"history": history, "thread_id": thread_id})


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clears conversation history for the current thread and session."""
    session_id = session.get("session_id")
    thread_id = session.get("thread_id")
    session.pop("history", None)

    try:
        if thread_id:
            db.clear_session_history(session_id, thread_id=thread_id)
        elif session_id:
            db.clear_session_history(session_id)
    except Exception as db_err:
        print("DB clear warning:", repr(db_err))

    return jsonify({"message": "Conversation history cleared."})


@app.route("/export", methods=["GET"])
def export_chat():
    """Exports conversation as downloadable Markdown."""
    session_id = session.get("session_id")
    thread_id = session.get("thread_id")
    doc_entry = documents.get(session_id) if session_id else None
    doc_name = doc_entry.get("metadata", {}).get("filename") if doc_entry else None

    history = []
    thread_title = "Helix AI Conversation"
    if thread_id:
        t_info = db.get_thread(thread_id)
        if t_info:
            thread_title = t_info.get("title", thread_title)
        try:
            history = db.get_history_as_turns(thread_id=thread_id, limit=500)
        except Exception:
            pass

    if not history and session_id:
        try:
            history = db.get_history_as_turns(session_id=session_id, limit=500)
        except Exception:
            pass

    if not history:
        history = session.get("history", [])

    md_lines = [
        f"# Helix AI Conversation Export",
        f"Thread: {thread_title}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    ]
    if doc_name:
        md_lines.append(f"**Referenced Document:** {doc_name}\n")
    md_lines.append("---\n")

    if not history:
        md_lines.append("*No messages in this conversation.*")
    else:
        for turn in history:
            u_text = turn.get("user", "")
            a_text = turn.get("assistant", "")
            t_stamp = turn.get("timestamp", "")
            time_label = f" ({t_stamp})" if t_stamp else ""
            md_lines.append(f"### 👤 You{time_label}:\n{u_text}\n")
            md_lines.append(f"### 🧬 Helix AI:\n{a_text}\n")
            md_lines.append("---\n")

    content = "\n".join(md_lines)
    return Response(
        content,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=helix_ai_{int(time.time())}.md"}
    )


# ─────────────────────────── SQL Studio API Endpoints ──────────────────────

@app.route("/api/sql/tables", methods=["GET"])
def api_sql_tables():
    """Returns metadata for all tables in the SQLite database."""
    try:
        tables = db.get_all_tables_info()
        return jsonify({"success": True, "tables": tables})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sql/query", methods=["POST"])
def api_sql_query():
    """Executes a custom SQL query and returns column headers and rows."""
    try:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "")
        max_rows = min(int(data.get("max_rows", 200)), 500)

        result = db.execute_user_query(query, max_rows=max_rows)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sql/import-csv", methods=["POST"])
def api_sql_import_csv():
    """Imports CSV data into a new or replaced SQLite table."""
    try:
        table_name = request.form.get("table_name", "").strip()

        csv_text = ""
        if "file" in request.files:
            file = request.files["file"]
            if not table_name:
                table_name = os.path.splitext(file.filename)[0]
            csv_text = file.read().decode("utf-8", errors="replace")
        else:
            json_data = request.get_json(silent=True) or {}
            table_name = table_name or json_data.get("table_name", "")
            csv_text = json_data.get("csv_data", "")

        if not csv_text.strip():
            return jsonify({"success": False, "error": "No CSV content provided."}), 400

        result = db.import_csv_to_table(table_name or "imported_table", csv_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sql/stats", methods=["GET"])
def api_sql_stats():
    """Returns database metrics, table count, file size, and record totals."""
    try:
        return jsonify(db.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sql/export", methods=["GET"])
def api_sql_export():
    """Downloads the raw SQLite database file."""
    try:
        db_path = db.DATABASE_PATH
        if os.path.exists(db_path):
            return send_file(
                db_path,
                as_attachment=True,
                download_name="helix_database.db",
                mimetype="application/x-sqlite3"
            )
        return jsonify({"error": "Database file not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """Returns database statistics for monitoring."""
    try:
        return jsonify(db.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/db/history", methods=["GET"])
def db_full_history():
    """Returns full persisted history for the current thread or session from the DB."""
    thread_id = request.args.get("thread_id") or session.get("thread_id")
    session_id = session.get("session_id")
    try:
        history = []
        if thread_id:
            history = db.get_history_as_turns(thread_id=thread_id, limit=500)
        elif session_id:
            history = db.get_history_as_turns(session_id=session_id, limit=500)
        return jsonify({"history": history, "thread_id": thread_id, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File size exceeds maximum allowed limit (25 MB)."}), 413


if __name__ == "__main__":
    db.init_db()
    port = int(os.getenv("PORT", 5000))
    print(f"Starting Helix AI server on http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
