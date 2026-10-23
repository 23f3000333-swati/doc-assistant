"""Web app: upload documents, then chat with an assistant that answers only from them.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

import streamlit as st

from doc_assistant import Response, SupportAssistant
from doc_assistant.config import KNOWLEDGE_BASE_DIR
from doc_assistant.loader import load_files, load_knowledge_base
from doc_assistant.retriever import load_embedder, tokenize

st.set_page_config(page_title="Document Support Assistant", layout="wide", initial_sidebar_state="expanded")

# ============================================================================ styling

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #f6f7f9; --surface: #ffffff; --border: #e4e7ec; --text: #101828; --muted: #667085;
  --accent: #1d4ed8; --accent-soft: #eef2ff; --mark: #fdf0c4;
  --ok: #067647; --ok-soft: #ecfdf3; --warn: #b54708; --warn-soft: #fffaeb;
  --neutral-soft: #f2f4f7; --violet: #5925dc; --violet-soft: #f4f3ff;
}
html, body, [class*="css"], .stApp, button, input, textarea { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

/* Main column */
.block-container, [data-testid="stMainBlockContainer"] { max-width: 860px; padding-top: 2.2rem; padding-bottom: 7rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.6rem; }
.side-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); margin: 1.4rem 0 0.6rem; }
.side-title:first-child { margin-top: 0; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0.4rem 0 0.8rem; }
.stats.two { grid-template-columns: repeat(2, 1fr); }
.stat { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; background: var(--surface); }
.stat .v { font-size: 1.25rem; font-weight: 600; color: var(--text); line-height: 1.2; }
.stat .l { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }
.kv { font-size: 0.84rem; color: var(--muted); margin: 0.35rem 0; }
.kv b { color: var(--text); font-weight: 600; }
.notice { font-size: 0.82rem; color: var(--muted); background: var(--neutral-soft); border-radius: 8px;
  padding: 10px 12px; margin: 0.6rem 0; line-height: 1.45; }
.history { margin: 0.3rem 0 0; padding-left: 1.1rem; font-size: 0.84rem; color: var(--text); }
.history li { margin: 0.25rem 0; }
.section-list { font-size: 0.82rem; margin: 0.2rem 0 0.8rem; }
.section-list .doc { font-weight: 600; color: var(--text); margin: 0.5rem 0 0.2rem; word-break: break-all; }
.section-list .sec { color: var(--muted); padding: 1px 0 1px 10px; border-left: 2px solid var(--border); }

/* Page header */
.page-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text); margin: 0; }
.page-sub { color: var(--muted); font-size: 0.95rem; margin: 0.3rem 0 1.6rem; }
.empty { border: 1px dashed var(--border); border-radius: 12px; padding: 22px 24px; background: var(--surface); }
.empty h4 { margin: 0 0 6px; font-size: 1rem; font-weight: 600; color: var(--text); }
.empty p { margin: 0 0 12px; color: var(--muted); font-size: 0.9rem; }

/* Messages */
.msg-user { display: flex; justify-content: flex-end; margin: 18px 0 10px; }
.msg-user div { background: var(--text); color: #fff; padding: 10px 14px; border-radius: 12px 12px 2px 12px;
  max-width: 78%; font-size: 0.95rem; line-height: 1.5; }
.msg-bot { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 18px 14px; margin: 0 0 8px; max-width: 92%; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.badge { display: inline-block; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 999px; margin-bottom: 10px; }
.b-answer { background: var(--accent-soft); color: var(--accent); }
.b-follow { background: var(--violet-soft); color: var(--violet); }
.b-switch { background: var(--ok-soft); color: var(--ok); }
.b-repeat, .b-scope { background: var(--warn-soft); color: var(--warn); }
.b-neutral { background: var(--neutral-soft); color: var(--muted); }
.note { font-size: 0.85rem; color: var(--muted); margin: 0 0 8px; line-height: 1.5; }
.note b { color: var(--text); font-weight: 600; }
.body { font-size: 0.97rem; line-height: 1.65; color: var(--text); }
.body strong { font-weight: 600; }
.body mark { background: var(--mark); color: inherit; padding: 0; border-radius: 2px; box-shadow: 0 0 0 2px var(--mark); }
.hint { font-size: 0.82rem; color: var(--muted); margin-top: 8px; }
.source { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 14px; padding-top: 12px;
  border-top: 1px solid var(--border); font-size: 0.82rem; }
.source .tag { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.source .ref { background: var(--neutral-soft); border-radius: 6px; padding: 3px 8px; color: var(--muted); }
.source .ref b { color: var(--text); font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.chip.more { color: var(--muted); background: var(--neutral-soft); border-color: transparent; }
.chip { font-size: 0.78rem; border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px;
  color: var(--text); background: var(--surface); }
.recall { margin: 4px 0 0; padding-left: 1.2rem; }
.recall li { margin: 6px 0; font-size: 0.93rem; }
.recall .q { font-weight: 500; }
.recall .s { color: var(--muted); font-size: 0.82rem; }

/* Chat input */
[data-testid="stChatInput"] textarea { font-size: 0.95rem; }
[data-testid="stBottom"] > div { background: var(--bg); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================================ data


@st.cache_resource(show_spinner="Loading the language model (first run only)...")
def shared_embedder():
    """Load the sentence-transformers model once and share it between sessions."""
    return load_embedder("auto")


def build_assistant(uploads) -> SupportAssistant:
    """Read the uploaded files (or the sample documents) and build a fresh assistant."""
    if uploads:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in uploads:
                path = Path(tmp) / f.name
                path.write_bytes(f.getvalue())
                paths.append(path)
            kb = load_files(paths, names=[f.name for f in uploads])
    else:
        kb = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    return SupportAssistant(kb=kb, embedder=shared_embedder())


# ============================================================================ formatting helpers

UNITS = {"rupees", "percent", "gb", "tb", "mb", "day", "days", "hour", "hours", "minute", "minutes", "week",
         "weeks", "month", "months", "year", "years", "am", "pm", "times", "working", "books", "attempts"}
e = html.escape


def highlight(text: str, question: str) -> str:
    """Escape text; bold numbers (with their unit) and mark the first match of each question word."""
    keys = set(tokenize(question))
    marked: set[str] = set()
    word_re = r"[A-Za-z0-9@%_-]+(?:[.,][A-Za-z0-9@%_-]+)*"
    parts = re.findall(rf"{word_re}|[^A-Za-z0-9@%_-]+", text)
    out, i = [], 0
    while i < len(parts):
        word = parts[i]
        if re.fullmatch(r"[0-9][0-9,.%]*", word):
            phrase = word
            j = i + 1
            while j + 1 < len(parts) and parts[j] == " " and parts[j + 1].lower() in UNITS:
                phrase += " " + parts[j + 1]
                j += 2
            out.append(f"<strong>{e(phrase)}</strong>")
            i = j
            continue
        stem = tokenize(word)
        if stem and stem[0] in keys and stem[0] not in marked and len(word) > 2:
            marked.add(stem[0])
            out.append(f"<mark>{e(word)}</mark>")
        else:
            out.append(e(word))
        i += 1
    return "".join(out)


def emphasise_note(note: str) -> str:
    """Bold section references and turn numbers inside a context note."""
    s = e(note)
    s = re.sub(r"(§[\d.]+(?: [A-Z][\w-]*(?: (?:and |or |of |to )?[A-Z][\w-]*)*)?)", r"<b>\1</b>", s)
    s = re.sub(r"\b(turn \d+)", r"<b>\1</b>", s)
    return s


def source_html(citations: list[str]) -> str:
    refs = []
    for c in citations:
        doc, _, section = c.partition(" › ")
        refs.append(f'<span class="ref">{e(doc)} &nbsp;›&nbsp; <b>{e(section)}</b></span>')
    return f'<div class="source"><span class="tag">Source</span>{"".join(refs)}</div>'


def badge(kind: str, note: str) -> str:
    if kind == "follow_up":
        return '<span class="badge b-follow">Follow-up</span>'
    if kind == "answer":
        if note.startswith("Switching topics"):
            return '<span class="badge b-switch">Topic switch</span>'
        if note.startswith("Moving on"):
            return '<span class="badge b-switch">New section</span>'
        return '<span class="badge b-answer">Answer</span>'
    return {
        "repeat": '<span class="badge b-repeat">Already answered</span>',
        "out_of_scope": '<span class="badge b-scope">Not in documents</span>',
        "recall": '<span class="badge b-neutral">Session history</span>',
    }.get(kind, "")


def render_answer(r: Response, question: str) -> str:
    main = r.text.split("\n\nSource: ")[0]
    lines = main.split("\n")
    note, body, hint, extra = "", main, "", ""

    if r.kind in ("answer", "follow_up") and len(lines) > 1:
        note, body = lines[0], " ".join(lines[1:])
    elif r.kind == "repeat":
        m = re.match(r"(You already asked this in turn \d+), so here's a short recap instead of repeating "
                     r"everything: (.*?)\n(.*)", main, re.S)
        if m:
            note, body, hint = m.group(1) + ".", m.group(2), m.group(3)
    elif r.kind == "out_of_scope":
        msg, _, topics = main.partition(" I can help with: ")
        body = msg
        topics = topics.rstrip(".")
        more = re.search(r" and (\d+) more$", topics)
        if more:
            topics = topics[: more.start()]
        chips = [f'<span class="chip">{e(t.strip())}</span>' for t in topics.split(", ") if t.strip()]
        if more:
            chips.append(f'<span class="chip more">+{more.group(1)} more</span>')
        extra = f'<div class="chips">{"".join(chips)}</div>'
    elif r.kind == "recall":
        items = []
        for line in lines[1:]:
            m = re.match(r'- Turn (\d+): "(.*)"( \(repeat\))? → (.*)', line)
            if m:
                rep = " (repeat)" if m.group(3) else ""
                items.append(f'<li><span class="q">{e(m.group(2))}</span>{e(rep)}<br>'
                             f'<span class="s">Turn {m.group(1)} · {e(m.group(4))}</span></li>')
        note, body = lines[0], ""
        extra = f'<ol class="recall">{"".join(items)}</ol>'

    parts = [badge(r.kind, note)]
    if note:
        parts.append(f'<div class="note">{emphasise_note(note)}</div>')
    if body:
        text = highlight(body, question) if r.kind in ("answer", "follow_up", "repeat") else e(body)
        parts.append(f'<div class="body">{text}</div>')
    if hint:
        parts.append(f'<div class="hint">{e(hint)}</div>')
    parts.append(extra)
    if r.citations:
        parts.append(source_html(r.citations))
    return f'<div class="msg-bot">{"".join(parts)}</div>'


# ============================================================================ sidebar

with st.sidebar:
    st.markdown('<div class="side-title">Documents</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Upload PDF, Word, TXT or Markdown files", type=["pdf", "docx", "txt", "md"],
                               accept_multiple_files=True)
    signature = tuple((f.name, f.size) for f in uploads) if uploads else ("sample",)

    if st.session_state.get("signature") != signature:
        try:
            with st.spinner("Reading and indexing documents..."):
                st.session_state.bot = build_assistant(uploads)
            st.session_state.signature = signature
            st.session_state.messages = []
        except Exception as exc:
            st.error(f"Could not read the documents: {exc}")
            st.stop()

    bot: SupportAssistant = st.session_state.bot
    kb = bot.kb
    if not uploads:
        st.markdown('<div class="notice">No files uploaded. The sample CloudNest documents are loaded; '
                    'upload your own files to replace them.</div>', unsafe_allow_html=True)

    n_sections = len({s.section_key for s in kb.sentences})
    st.markdown(f'<div class="stats"><div class="stat"><div class="v">{len(kb.documents)}</div>'
                f'<div class="l">Documents</div></div><div class="stat"><div class="v">{n_sections}</div>'
                f'<div class="l">Sections</div></div><div class="stat"><div class="v">{len(kb.sentences)}</div>'
                f'<div class="l">Sentences</div></div></div>', unsafe_allow_html=True)

    with st.expander("Indexed sections"):
        rows = []
        for doc in kb.documents:
            labels = list(dict.fromkeys(s.label for s in kb.sentences if s.doc == doc))
            rows.append(f'<div class="doc">{e(doc)}</div>' + "".join(f'<div class="sec">{e(l)}</div>' for l in labels))
        st.markdown(f'<div class="section-list">{"".join(rows)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-title">Session</div>', unsafe_allow_html=True)
    turns = bot.memory.content_turns()
    st.markdown(f'<div class="stats two"><div class="stat"><div class="v">{len(turns)}</div>'
                f'<div class="l">Questions answered</div></div><div class="stat"><div class="v">'
                f'{len(bot.memory.shown)}</div><div class="l">Sentences shown</div></div></div>',
                unsafe_allow_html=True)
    topic = e(bot.memory.current.label) if bot.memory.current else "None yet"
    st.markdown(f'<div class="kv">Current topic: <b>{topic}</b></div>', unsafe_allow_html=True)
    if turns:
        items = "".join(f"<li>{e(t.question)}</li>" for t in turns)
        st.markdown(f'<div class="kv">Asked so far</div><ol class="history">{items}</ol>', unsafe_allow_html=True)
    retrieval = "Semantic (MiniLM) + keyword" if bot.backend == "sentence-transformers" else "Keyword (TF-IDF) + BM25"
    st.markdown(f'<div class="kv">Retrieval: <b>{retrieval}</b></div>', unsafe_allow_html=True)
    if st.button("Reset conversation", use_container_width=True):
        bot.reset()
        st.session_state.messages = []
        st.rerun()

# ============================================================================ main

st.markdown('<p class="page-title">Document Support Assistant</p>'
            '<p class="page-sub">Answers come only from your documents, with the exact section they were taken '
            'from. The assistant remembers the conversation and does not repeat itself.</p>',
            unsafe_allow_html=True)

if not st.session_state.messages:
    chips = "".join(f'<span class="chip">{e(t)}</span>' for t in dict.fromkeys(kb.doc_titles.values()))
    st.markdown(f'<div class="empty"><h4>Ready</h4><p>Ask a question about the loaded documents below. '
                f'To use your own files, upload them in the sidebar.</p><div class="chips">{chips}</div></div>',
                unsafe_allow_html=True)

for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f'<div class="msg-user"><div>{e(m["text"])}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(render_answer(m["response"], m["question"]), unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "text": prompt})
    st.session_state.messages.append({"role": "assistant", "response": bot.ask(prompt), "question": prompt})
    st.rerun()