# Helix AI — Intelligent RAG Chatbot & SQL Database Studio

An advanced AI-powered conversational assistant with **Neural Retrieval-Augmented Generation (RAG)**, integrated **SQLite Relational Database Studio**, multi-thread conversation management, and **Google Gemini API** integration.

---

## 📌 Project Overview

Helix AI provides a complete conversational AI experience with enterprise-grade document intelligence and SQL relational persistence:
- **Application Framework**: Developed using Google Antigravity & Python Flask.
- **LLM Integration**: Integrated Google Gemini API (`google-genai` SDK) with automatic model fallback (`gemini-flash-lite-latest`, `gemini-flash-latest`, `gemini-3.8-flash`, `gemini-pro-latest`).
- **SQL Relational Database Studio**: Built-in SQLite persistence layer with a visual SQL Database Studio modal, Table Explorer, schema viewer, interactive SQL query editor with CSV download, and automated CSV-to-table importer.
- **Natural Language Text-to-SQL & `/sql` Command**: Query the live SQLite database in chat using natural language or direct `/sql <query>` slash commands with auto-rendered interactive Markdown tables.
- **Multi-Thread Conversation Management**: Persistent conversation threads organized in the sidebar with create, switch, rename, and delete capabilities.
- **Document QA & Neural RAG**: TF-IDF semantic vector retrieval over uploaded PDF, TXT, MD, CSV, JSON, and code files.
- **User-Friendly Modern UI**: Dark & Light mode theme toggle, voice speech recognition input, text-to-speech read-aloud, keyboard shortcuts helper (`Ctrl+K`, `Ctrl+/`), code block copy buttons, and drag-and-drop file ingestion.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Browser                        │
│   (HTML5, CSS3, Plus Jakarta Sans, marked.js, highlight.js) │
│   - Multi-Thread Sidebar     - SQL Database Studio Modal   │
│   - Dark / Light Theme       - Audio & Keyboard Shortcuts   │
└───────────────┬─────────────────────────────▲───────────────┘
                │ HTTP POST /chat, /upload    │ JSON / SSE Response
                ▼                             │
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                   │
├──────────────────────────────┬──────────────────────────────┤
│      Document Ingestion      │     Conversation Manager     │
│  - PyPDF & Stream Extraction │  - Threaded SQL Persistence  │
│  - Chunking with Overlap     │  - Context Augmentation      │
│  - TF-IDF Vectorization      │  - Error Handling & Fallback │
│  - Cosine Similarity Search  │  - /sql Command Execution    │
├──────────────────────────────┴──────────────────────────────┤
│                SQLite Engine (database.py)                  │
│  - Threads, Sessions, Chat History, Document Metadata       │
│  - Custom Tables, CSV Importer, Schema Introspection        │
└───────────────┬─────────────────────────────▲───────────────┘
                │                             │
                ▼                             │
┌─────────────────────────────────────────────────────────────┐
│                      Google Gemini API                      │
│        (gemini-flash-lite-latest / gemini-3.8-flash)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

1. **Integrated SQL Database Studio**:
   - **Table Explorer**: Inspect all SQLite tables (`threads`, `chat_history`, `documents`, `sessions`, and imported user tables) with schemas, column types, and interactive data grids.
   - **Query Studio**: Run custom SQL queries (`SELECT`, `PRAGMA`, etc.) with quick-run presets and download results as CSV.
   - **CSV to SQL Importer**: Upload spreadsheets to create live SQLite tables on the fly.
   - **Raw Database Backup**: One-click download of the active SQLite `.db` database file.
   - **Direct `/sql` Command**: Type `/sql SELECT * FROM threads;` in the chat input for instant query results.

2. **Persistent Multi-Thread Conversations**:
   - Conversations are saved in the SQLite database across browser restarts.
   - Sidebar list with real-time filtering, instant switching, renaming, and safe deletion.
   - Auto-generated thread titles based on user intent.

3. **Dual Intelligence Modes (Chat & Document RAG)**:
   - **General AI Assistant**: Code generation, reasoning, creative writing, and system architecture.
   - **Document QA (RAG)**: Ask targeted questions about uploaded PDFs, CSV datasets, or codebases.
   - In-memory PDF parsing and multi-encoding code reading (`.pdf`, `.txt`, `.md`, `.csv`, `.json`, `.py`, `.js`, `.sql`).

4. **User-Friendly Experience & Ergonomics**:
   - **Dark & Light Mode Toggle**: Seamlessly switch between obsidian cyber-dark and clean-slate light themes.
   - **Voice Input**: Speech-to-text dictation using the Web Speech API.
   - **Audio Playback**: One-click text-to-speech read aloud for AI responses.
   - **Interactive Code Blocks**: Syntax highlighting with one-click copy buttons.
   - **Keyboard Shortcuts**: `Ctrl+K` (New thread), `Ctrl+/` (SQL Studio), `Enter` (Send), `Esc` (Close modal).

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.13, Flask | REST API routing, session management, thread coordination |
| **Database** | SQLite 3 (`database.py`) | Relational persistence for threads, messages, docs, and custom data |
| **AI / LLM** | Google Gemini API (`google-genai` SDK) | Natural language understanding, generation & Text-to-SQL |
| **Document Processing** | `pypdf` | In-memory PDF text extraction |
| **RAG / Vectorization** | `scikit-learn` (TF-IDF, Cosine Similarity) | Semantic chunk retrieval for attached documents |
| **Frontend** | HTML5, Modern CSS3, JavaScript (ES6) | Responsive UI, SQL Studio modal, multi-thread management |
| **Formatting & Highlighting** | `marked.js`, `highlight.js`, `DOMPurify` | Markdown rendering, code syntax highlighting, XSS sanitization |

---

## ⚙️ Installation & Setup

### 1. Open Project Directory
```powershell
cd D:\gradtwin\AI_Chatbot_Project
```

### 2. Configure Environment Variables
Verify `.env` in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```powershell
python -m unittest discover -s . -p "test_*.py"
```
*(All 16 automated unit tests should pass).*

### 5. Launch Application
```powershell
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 📡 API Endpoints

### Core Chat & Document Routes
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page chat & studio interface |
| `GET` | `/health` | Returns service status, database metrics, and active models |
| `POST` | `/chat` | Processes questions, handles `/sql` commands, and augments RAG context |
| `POST` | `/upload` | Uploads and indexes PDF/TXT/CSV files for RAG & SQL import |
| `GET` | `/document` | Returns metadata of currently attached document |
| `DELETE` | `/document` | Detaches active document |
| `GET` | `/history` | Retrieves conversation history for active thread |
| `POST` | `/clear` | Clears conversation history for active thread |
| `GET` | `/export` | Downloads the conversation as Markdown (`.md`) |

### Thread Management Routes
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/threads` | Returns list of all conversation threads |
| `POST` | `/threads` | Creates a new conversation thread |
| `POST` | `/threads/switch` | Switches active conversation thread |
| `PATCH` | `/threads/<id>` | Renames an existing conversation thread |
| `DELETE` | `/threads/<id>` | Permanently deletes a thread and its messages |

### SQL Database Studio Routes
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/sql/tables` | Returns list of all SQLite tables, schemas, and row counts |
| `POST` | `/api/sql/query` | Executes custom SQL query with timing and tabular results |
| `POST` | `/api/sql/import-csv` | Converts uploaded CSV into a new SQLite table |
| `GET` | `/api/sql/stats` | Returns database file size, record totals, and table metrics |
| `GET` | `/api/sql/export` | Downloads the raw SQLite `.db` database file |

---

## 🧪 Testing

The comprehensive test suite in `test_app.py` covers 16 test cases:
- `test_split_text`: Overlapping chunk generation.
- `test_extract_text_from_txt_file` & `test_extract_text_from_py_file`: File parsers.
- `test_build_document_index_and_retrieve`: TF-IDF matrix & semantic chunk retrieval.
- `test_upload_text_file` & `test_upload_unsupported_file`: Upload validation & security.
- `test_chat_validation`: Empty & invalid message checking.
- `test_history_and_clear_routes`: Conversation history lifecycle.
- `test_export_route`: Markdown chat export.
- `test_thread_management`: Multi-thread creation, switching, renaming, and deletion.
- `test_sql_studio_api`: Table introspection, custom query execution, and database stats.
- `test_sql_slash_command`: Direct `/sql` query execution through the chat API.
- `test_sql_csv_import`: CSV spreadsheet to SQLite table conversion and querying.
