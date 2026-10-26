# Document Assistant

A support assistant that answers questions **only from the documents you upload**. It remembers the conversation, avoids repeating itself, handles topic changes, and cites the exact document section in every answer. It runs fully on your own computer: no API key, no paid service.

---

## Problem Statement

> Using your own documents as a knowledge base, build a document-aware support assistant that maintains context across a multi-turn conversation. The assistant should: remember what the user has already asked within the session, avoid repeating information it has already provided, gracefully handle topic switches, and cite the specific document section it is drawing from in each response. Demonstrate it across a conversation of at least 10 turns.

---

## Solution

The uploaded documents are split into sections and sentences. For each question, a **hybrid search** (meaning + keywords) finds the best matching sentences, and a **session memory** decides how to reply.

| Requirement | How it is solved |
|---|---|
| Own documents as knowledge base | PDF, Word, TXT and Markdown files are uploaded and read at run time |
| Context across turns | Follow-ups like "tell me more" or "what if…" use the current topic from memory |
| Remembers what was asked | Repeated questions are detected; "What have I asked so far?" lists the session |
| Avoids repeating information | Sentences already shown are tracked and skipped in later answers |
| Handles topic switches | A change of section or document is detected and announced |
| Cites the section | Every answer ends with its source, e.g. `billing_policy.md › §2.3 Refund Policy` |

Answers are taken word for word from the documents, so the assistant does not make up information. If something is not in the documents, it says so.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Programming language |
| **Streamlit** | Web interface (upload, chat, session panel) |
| **sentence-transformers** | Semantic search: converts text into embeddings |
| **all-MiniLM-L6-v2** | Pre-trained embedding model (384-dimension vectors, ~90 MB) |
| **PyTorch** | Runs the embedding model |
| **NumPy** | Stores the vectors and computes cosine similarity |
| **BM25** (own implementation) | Keyword search for exact terms |
| **TF-IDF** (own implementation) | Fallback search if the model is not available |
| **pypdf** | Reads PDF files |
| **python-docx** | Reads Word (.docx) files |
| **unittest** | Automated tests (16 tests) |

---

## Pipeline

**1. Indexing (once, when documents are uploaded)**

```
Documents (PDF / DOCX / TXT / MD)
   → Extract text            (pypdf, python-docx)
   → Detect sections         (headings such as "2.3 Refund Policy")
   → Split into sentences    (each keeps its document + section for citation)
   → Create embeddings       (all-MiniLM-L6-v2) + build BM25 keyword index
```

**2. Answering (every question)**

```
Question
   → Embed the question
   → Hybrid search:  score = 0.7 × semantic similarity + 0.3 × BM25 keyword score
   → Relevance check (below threshold → "not in the documents")
   → Pick the best section, remove sentences already shown
   → Answer + source section
   → Save the turn in session memory
```

**Session memory** (`SessionMemory` class, kept for the browser session):
- `turns`: every question with its embedding and the sentences shown
- `shown`: IDs of sentences already shown, and in which turn
- `current`: the current topic (document and section)

---

## Workflow

For every message, the assistant decides how to reply in this order:

1. **Small talk:** greetings and thanks get a short reply.
2. **Session recall:** "What have I asked so far?" lists all previous questions.
3. **Repeat check:** if the question was already asked, it gives a one-line recap and points to the earlier turn.
4. **Follow-up check:** vague questions like "tell me more" are combined with the current topic.
5. **Search:** hybrid search finds the most relevant sentences.
6. **Out of scope:** if nothing is relevant, it says so instead of guessing.
7. **No repetition:** sentences already shown are skipped.
8. **Topic switch:** if the answer comes from a new section or document, it says "Switching topics".
9. **Reply:** up to 3 new sentences with the source section, and the turn is saved to memory.

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/doc-support-assistant.git
cd doc-support-assistant
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
# Linux only (recommended): CPU version of PyTorch, avoids a ~4 GB GPU download
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All systems
pip install -r requirements.txt
```

**4. Start the app**

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser. The embedding model (~90 MB) downloads once on the first run.

**5. Use it**

Upload your documents in the sidebar (or use the sample documents loaded by default), then ask questions in the chat.

**Run the tests (optional)**

```bash
python -m unittest discover -s tests -v
```
