# Helix AI — Project Presentation Slides (Markdown Deck)

## Slide 1: Title & Introduction
* **Project Title**: Helix AI — Intelligent Chatbot & RAG Assistant
* **Context**: Internship Capstone Project Assignment
* **Environment**: Developed using Google Antigravity & Python
* **Core Technologies**: Flask, Google Gemini API, TF-IDF Vector Retrieval, Modern Web Frontend
* **Objective**: Build a production-ready conversational AI application capable of general dialogue and document-based question answering (RAG).

---

## Slide 2: Project Requirements & Scope
* **Application Architecture**: Decoupled full-stack architecture with Flask backend and responsive vanilla JS/CSS frontend.
* **AI Model Integration**: Integrated Google Gemini API using Google AI Studio API key with multi-model fallback chain (`gemini-flash-lite-latest`, `gemini-flash-latest`, `gemini-3.8-flash`, `gemini-pro-latest`).
* **Conversation Memory**: Maintained multi-turn chat history in session for contextual follow-up questions.
* **Error Resilience**: User-friendly handling for quota limits (429), model unavailability (503/404), authentication errors (401), and invalid inputs.
* **Document QA (RAG)**: Ingestion and semantic retrieval over PDF, text, and code files.

---

## Slide 3: System Architecture & Workflow
* **Client Tier**: Modern single-page interface with real-time markdown parsing, syntax highlighting, and drag-and-drop file ingestion.
* **Server Tier (Flask)**:
  * Route dispatching (`/chat`, `/upload`, `/document`, `/history`, `/clear`, `/export`).
  * In-memory session store with automated document pruning.
* **Retrieval Tier**:
  * In-memory stream parsing via `pypdf`.
  * Multi-encoding text support (`utf-8`, `latin-1`).
  * Overlapping sliding-window chunking (400 words, 80-word overlap).
  * TF-IDF matrix generation and cosine similarity calculation.
* **AI Generation Tier**: Google Gemini API called with structured system instructions and context-augmented prompts.

---

## Slide 4: Retrieval-Augmented Generation (RAG) Deep Dive
* **Why RAG?**: Large Language Models have a static knowledge cutoff and cannot access private documents without hallucinating. RAG anchors generation in verifiable document context.
* **The Indexing Pipeline**:
  1. *Upload*: File read directly as an in-memory stream without writing temporary files to disk.
  2. *Chunking*: Text broken into overlapping chunks to preserve semantic integrity across boundaries.
  3. *Vectorization*: TF-IDF vectorizer extracts unigrams and bigrams with sublinear term-frequency scaling.
  4. *Query Matching*: When a user asks a question, cosine similarity matches top-k chunks.
  5. *Dual-Mode Context Injection*: Small-to-medium files (<35,000 characters) are injected in full; larger documents combine an introductory overview with top semantic excerpts.

---

## Slide 5: Frontend Design & User Experience
* **Design Philosophy**: High-contrast, distraction-free dark interface inspired by modern tools like Linear, Claude, and ChatGPT.
* **Key Features**:
  * Rich Markdown typography with code copy buttons.
  * Attached document indicator chips in topbar, sidebar, and dock.
  * Quick-prompt chips for instant summaries and key takeaway generation.
  * One-click conversation export to Markdown (`.md`).
  * Fullscreen drag-and-drop dropzone.

---

## Slide 6: Verification & Testing
* **Automated Test Suite (`test_app.py`)**:
  * 12 automated unit and integration tests covering chunk splitting, text decoding, TF-IDF ranking, upload validation, route dispatching, and markdown export.
  * 100% test pass rate with sub-second execution.

---

## Slide 7: Live Demonstration Walkthrough
1. **General Chat**: Ask conversational, coding, or concept explanation queries.
2. **Follow-Up Questions**: Demonstrate conversation memory across turns.
3. **Document Ingestion**: Drag and drop a PDF or text file into the interface.
4. **Targeted Document QA**: Ask specific questions and request comprehensive document summaries.
5. **Export**: Download the conversation history transcript.
