"""Read documents (.md, .txt, .pdf, .docx) and split them into sections and sentences.

Every sentence remembers which document and section it came from, so every answer can be cited.

How sections are found (in this order):
  1. Headings: Markdown "## 2.3 Refund Policy", Word "Heading" styles, or plain lines like "2.3 Refund Policy".
     Numbered headings are cited as "§2.3 Refund Policy"; unnumbered ones by their title.
  2. PDFs without headings are split by page   -> cited as "Page 3".
  3. Other files without headings are split into parts of a few sentences -> cited as "Part 2".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}

MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(.+)$")
PLAIN_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z][^.!?]{1,70})$")  # "2.3 Refund Policy"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
SENTENCES_PER_PART = 5


@dataclass(frozen=True)
class Sentence:
    id: int
    text: str
    doc: str            # file name, e.g. "billing_policy.md"
    doc_title: str      # document title, e.g. "CloudNest Billing Policy"
    section: str        # "2.3", or "" when the section has no number
    section_title: str  # "Refund Policy", "Page 3", "Part 2"

    @property
    def section_key(self) -> tuple[str, str, str]:
        return (self.doc, self.section, self.section_title)

    @property
    def label(self) -> str:
        return f"§{self.section} {self.section_title}" if self.section else self.section_title

    @property
    def citation(self) -> str:
        return f"{self.doc} › {self.label}"

    @property
    def search_text(self) -> str:
        # The section title gives each sentence context ("Refund Policy. Approved refunds return...").
        return f"{self.section_title}. {self.text}"


@dataclass
class KnowledgeBase:
    sentences: list[Sentence] = field(default_factory=list)

    def section_sentences(self, key) -> list[Sentence]:
        return [s for s in self.sentences if s.section_key == key]

    @property
    def doc_titles(self) -> dict[str, str]:
        return {s.doc: s.doc_title for s in self.sentences}

    @property
    def documents(self) -> list[str]:
        return list(dict.fromkeys(s.doc for s in self.sentences))

    def add_section(self, doc: str, doc_title: str, number: str, title: str, lines: list[str]) -> None:
        for line in lines:
            line = re.sub(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+", "", line).strip()   # strip bullets / list numbers
            for text in SENTENCE_SPLIT.split(line):
                text = text.strip()
                if len(text) > 2:
                    self.sentences.append(Sentence(len(self.sentences), text, doc, doc_title, number, title))


# ----------------------------------------------------------------------------- reading files

def _blocks(path: Path) -> list[tuple[str, str]]:
    """Return the file as (kind, text) blocks. kind: 'h<level>', 'p', or 'page' (PDF page break)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        blocks = []
        for page in PdfReader(str(path)).pages:
            blocks.append(("page", ""))
            blocks += [("p", line) for line in (page.extract_text() or "").splitlines()]
        return blocks
    if ext == ".docx":
        import docx

        blocks = []
        for para in docx.Document(str(path)).paragraphs:
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style == "title":
                blocks.append(("h1", para.text))
            elif style.startswith("heading"):
                level = re.findall(r"\d+", style)
                blocks.append((f"h{int(level[0]) + 1 if level else 2}", para.text))
            else:
                blocks.append(("p", para.text))
        return blocks

    blocks = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = MD_HEADING.match(line.strip()) if ext == ".md" else None
        blocks.append((f"h{len(m.group(1))}", m.group(2)) if m else ("p", line))
    return blocks


def _join_wrapped_lines(lines: list[str], is_pdf: bool) -> list[str]:
    """PDF text is broken at the end of every visual line; glue lines back into paragraphs.
    In a PDF a line not ending in . ! ? : continues on the next line; elsewhere only if the next starts lowercase."""
    out: list[str] = []
    for line in lines:
        if out and out[-1] and not re.search(r"[.!?:]$", out[-1]) and (is_pdf or line[:1].islower()):
            out[-1] += " " + line
        else:
            out.append(line)
    return out


def _next_is_heading(blocks: list[tuple[str, str]], index: int) -> bool:
    following = next(((k, t) for k, t in blocks[index + 1:] if t), None)
    return bool(following) and (following[0].startswith("h") or bool(PLAIN_HEADING.match(following[1])))


def add_document(kb: KnowledgeBase, path: str | Path, display_name: str | None = None) -> None:
    path = Path(path)
    doc = display_name or path.name
    blocks = [(k, t.strip()) for k, t in _blocks(path)]
    doc_title = Path(doc).stem.replace("_", " ").title()

    # The first top-level heading (or a Word "Title") names the document. Without one, a short
    # first line with no full stop (e.g. "Shipping FAQ" at the top of a PDF/TXT) is the title.
    first_text = next(((i, k, t) for i, (k, t) in enumerate(blocks) if t), None)
    titled = [t for k, t in blocks if k == "h1" and t]
    if titled:
        doc_title = titled[0]
    elif first_text and first_text[1].startswith("h") and _next_is_heading(blocks, first_text[0]):
        # A first heading with no text under it (e.g. Word "Heading 1" used as the title) names the document.
        doc_title = first_text[2]
        blocks[first_text[0]] = ("h1", first_text[2])
    elif (first_text and first_text[1] == "p" and len(first_text[2]) < 60
          and not re.search(r"[.!?:]$", first_text[2]) and not PLAIN_HEADING.match(first_text[2])):
        doc_title = first_text[2]
        blocks[first_text[0]] = ("h1", first_text[2])

    sections: list[tuple[str, str, list[str]]] = []   # (number, title, lines)
    current: tuple[str, str, list[str]] | None = None
    page = 0
    has_headings = any((k.startswith("h") and k != "h1") or (k == "p" and PLAIN_HEADING.match(t) and len(t) < 80)
                       for k, t in blocks)
    for kind, text in blocks:
        if kind == "page":
            page += 1
            if not has_headings:
                current = ("", f"Page {page}", [])
                sections.append(current)
            continue
        if not text or kind == "h1":
            continue
        heading = None
        if kind.startswith("h"):
            m = NUMBERED.match(text)
            heading = (m.group(1), m.group(2)) if m else ("", text)
        elif PLAIN_HEADING.match(text) and len(text) < 80:
            m = PLAIN_HEADING.match(text)
            heading = (m.group(1), m.group(2).strip())
        if heading:
            has_headings = True
            current = (heading[0], heading[1], [])
            sections.append(current)
            continue
        if current is None:
            current = ("", "Introduction", [])
            sections.append(current)
        current[2].append(text)

    # No headings at all (plain text / Word without heading styles): split into parts.
    if len(sections) == 1 and sections[0][1] == "Introduction":
        sentences = [s for line in sections[0][2] for s in SENTENCE_SPLIT.split(line) if s.strip()]
        sections = [("", f"Part {i // SENTENCES_PER_PART + 1}", sentences[i:i + SENTENCES_PER_PART])
                    for i in range(0, len(sentences), SENTENCES_PER_PART)]

    for number, title, lines in sections:
        kb.add_section(doc, doc_title, number, title, _join_wrapped_lines(lines, path.suffix.lower() == ".pdf"))


def load_files(paths, names=None) -> KnowledgeBase:
    kb = KnowledgeBase()
    for i, p in enumerate(paths):
        if Path(p).suffix.lower() in SUPPORTED:
            add_document(kb, p, names[i] if names else None)
    if not kb.sentences:
        raise ValueError("No readable text found in the documents.")
    return kb


def load_knowledge_base(folder: str | Path) -> KnowledgeBase:
    files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No documents ({', '.join(sorted(SUPPORTED))}) found in {folder}")
    return load_files(files)
