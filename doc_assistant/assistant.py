"""The support assistant: decides *how* to answer each turn using retrieval + memory.

Pipeline for every user message
-------------------------------
1. Small talk / "what can you do?"         -> fixed friendly reply
2. "What have I asked so far?"             -> answered from memory
3. Repeat detection                        -> short recap instead of the full answer again
4. Follow-up detection + query rewriting   -> vague questions inherit the current topic
5. Hybrid retrieval (semantic + BM25)      -> best matching sentences
6. Out-of-scope check                      -> refuse instead of guessing
7. Section choice + de-duplication         -> only sentences the user has not seen yet
8. Topic-transition note + citation        -> "Switching topics..." / "Source: file › §x.y Title"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .loader import KnowledgeBase, Sentence, load_knowledge_base
from .memory import SessionMemory, Turn
from .retriever import Hit, HybridRetriever, load_embedder, tokenize

FOLLOW_UP_PHRASES = ("tell me more", "what else", "anything else", "what about", "how about",
                     "what if", "and if", "more detail", "go on", "elaborate", "explain more")
PRONOUNS = {"it", "that", "this", "those", "them", "there", "its", "they"}

GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b[\s!.,]*$", re.I)
THANKS = re.compile(r"\b(thanks|thank you|thx|bye|goodbye)\b", re.I)
CAPABILITIES = re.compile(r"what can you (do|help)|what do you know|help me\??$", re.I)
RECALL = re.compile(r"what (have|did) i (ask|asked)|(summar(y|ise|ize)|recap) (of )?(our|this|the) "
                    r"(chat|conversation|session)|what have we (discussed|covered)", re.I)


@dataclass
class Response:
    text: str
    kind: str
    turn: int
    citations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


class SupportAssistant:
    def __init__(self, docs_dir=config.KNOWLEDGE_BASE_DIR, backend: str = "auto",
                 kb: KnowledgeBase | None = None, embedder=None):
        """Read the documents in `docs_dir` (or use a ready `kb`, e.g. from uploaded files).
        Pass an already-loaded `embedder` to avoid loading the model again."""
        self.kb = kb if kb is not None else load_knowledge_base(docs_dir)
        self.retriever = HybridRetriever(self.kb.sentences, embedder or load_embedder(backend))
        self.th = config.THRESHOLDS[self.retriever.backend]
        self.memory = SessionMemory()

    # ------------------------------------------------------------------ helpers
    @property
    def backend(self) -> str:
        return self.retriever.backend

    def _doc_name(self, doc: str) -> str:
        return self.kb.doc_titles.get(doc, doc)

    def _topics(self) -> str:
        labels = list(dict.fromkeys(s.section_title for s in self.kb.sentences))
        more = f" and {len(labels) - 12} more" if len(labels) > 12 else ""
        return ", ".join(labels[:12]) + more

    def _finish(self, question, kind, text, sentences=(), vector=None, cite=()) -> Response:
        """Save the turn to memory. `sentences` were shown to the user; `cite` are only cited."""
        turn = Turn(self.memory.turn_number, question, kind, vector, list(sentences))
        self.memory.add(turn)
        citations = list(dict.fromkeys(s.citation for s in [*sentences, *cite]))
        if citations:
            text += "\n\nSource: " + "; ".join(citations)
        return Response(text, kind, turn.number, citations)

    def _is_follow_up(self, question: str, own_hits: list[Hit]) -> bool:
        if self.memory.current is None:
            return False
        q = question.lower()
        if any(p in q for p in FOLLOW_UP_PHRASES) or not tokenize(question):
            return True
        has_pronoun = bool(set(re.findall(r"[a-z]+", q)) & PRONOUNS)
        return has_pronoun and own_hits[0].semantic < self.th["weak_query"]

    @staticmethod
    def _rank_sections(hits: list[Hit]) -> list[tuple[tuple[str, str], float]]:
        best: dict[tuple[str, str], float] = {}
        for h in hits:
            best[h.sentence.section_key] = max(best.get(h.sentence.section_key, -1.0), h.score)
        return sorted(best.items(), key=lambda kv: -kv[1])

    def _transition_note(self, chosen: Sentence, follow_up: bool) -> str:
        prev = self.memory.current
        if prev is None:
            return ""
        if chosen.section_key == prev.section_key:
            return f"Continuing with {prev.label}." if follow_up else ""
        if chosen.doc != prev.doc:
            return (f"Switching topics: from {self._doc_name(prev.doc)} "
                    f"({prev.label}) to {self._doc_name(chosen.doc)}.")
        return f"Moving on from {prev.label} to {chosen.label}."

    # ------------------------------------------------------------------ main entry point
    def ask(self, question: str) -> Response:
        q = question.strip()

        # 1. Small talk
        if GREETING.match(q):
            return self._finish(q, "smalltalk", f"Hello! I'm a document support assistant. I've read "
                                f"{len(self.kb.documents)} document(s): {', '.join(dict.fromkeys(self.kb.doc_titles.values()))}. "
                                f"Ask me anything about them.")
        if THANKS.search(q) and len(tokenize(q)) <= 3:
            answered = len(self.memory.content_turns())
            return self._finish(q, "smalltalk", f"You're welcome! We covered {answered} question(s) this "
                                "session. Feel free to come back any time.")
        if CAPABILITIES.search(q):
            return self._finish(q, "smalltalk", "I answer only from the documents I have been given. Topics I know: "
                                f"{self._topics()}.")

        # 2. Questions about the conversation itself
        if RECALL.search(q):
            return self._finish(q, "recall", self._recall_text())

        vector = self.retriever.embed(q)

        # 3. Repeated question
        previous, similarity = self.memory.most_similar_question(vector)
        if previous is not None and similarity >= self.th["repeat_question"]:
            recap = max(previous.sentences, key=lambda s: float(self.retriever.matrix[s.id] @ vector))
            text = (f"You already asked this in turn {previous.number}, so here's a short recap instead of "
                    f"repeating everything: {recap.text}\nSay \"tell me more\" if you want details I "
                    f"haven't covered yet.")
            return self._finish(q, "repeat", text, [recap], vector)

        # 4. Follow-up detection and query rewriting
        own_hits = self.retriever.search(q, top_k=None)
        follow_up = self._is_follow_up(q, own_hits)
        explicit = any(p in q.lower() for p in FOLLOW_UP_PHRASES) or not tokenize(q)
        if follow_up and not explicit:
            # Guard: a follow-up guessed only from a pronoun ("can I recover it?") is overridden
            # when the question on its own clearly points to a different section.
            own_key = self._rank_sections(own_hits)[0][0]
            if (own_key != self.memory.current.section_key
                    and own_hits[0].semantic >= self.th["clear_topic"]):
                follow_up = False
        hits = own_hits
        if follow_up:
            anchor = self.memory.current
            last = self.memory.last_content_turn()
            rewritten = f"{q} {anchor.section_title} {last.question if last else ''}"
            hits = self.retriever.search(rewritten, top_k=None)

        # 6. Out of scope
        if max(h.semantic for h in hits) < self.th["min_relevance"]:
            return self._finish(q, "out_of_scope", f"I couldn't find that in the documents, so I "
                                f"won't guess. I can help with: {self._topics()}.", vector=vector)

        # 7. Choose the section and the sentences the user has not seen yet
        sections = self._rank_sections(hits)
        best_key, top = sections[0]
        current = self.memory.current
        if follow_up and current is not None:
            current_score = dict(sections).get(current.section_key, -1.0)
            if current_score >= 0.6 * top:
                best_key = current.section_key

        notes: list[str] = []
        chosen = self._pick_sentences(hits, best_key, include_all=follow_up)
        skipped = [h for h in hits if h.sentence.section_key == best_key and self.memory.was_shown(h.sentence)]

        if not chosen:
            covered = self.memory.shown[skipped[0].sentence.id] if skipped else None
            first = self.kb.section_sentences(best_key)[0]
            notes.append(f"I already covered {first.label} in turn {covered}.")
            chosen = self._next_section(own_hits, exclude=best_key)
            if not chosen:
                text = notes[0] + " That's everything the documents say about it."
                return self._finish(q, "repeat", text, [], vector, cite=[first])
            notes.append(f"The related section {chosen[0].citation} says:")
        elif skipped:
            notes.append(f"(Leaving out what I already told you in turn {self.memory.shown[skipped[0].sentence.id]}.)")

        # 8. Transition note + answer + citation
        transition = self._transition_note(chosen[0], follow_up)
        if transition and not any("already covered" in n for n in notes):
            notes.insert(0, transition)
        body = " ".join(s.text for s in sorted(chosen, key=lambda s: s.id))
        text = (" ".join(notes) + "\n" if notes else "") + body
        return self._finish(q, "follow_up" if follow_up else "answer", text, chosen, vector)

    def _pick_sentences(self, hits: list[Hit], key, include_all: bool) -> list[Sentence]:
        in_section = [h for h in hits if h.sentence.section_key == key]
        if not in_section:
            return []
        floor = 0.0 if include_all else 0.45 * in_section[0].score
        fresh = [h.sentence for h in in_section if not self.memory.was_shown(h.sentence) and h.score >= floor]
        return fresh[: config.MAX_SENTENCES_PER_ANSWER]

    def _next_section(self, own_hits: list[Hit], exclude) -> list[Sentence]:
        top = own_hits[0].score
        for key, score in self._rank_sections(own_hits):
            if key == exclude or score < 0.4 * top:
                continue
            picked = [h.sentence for h in own_hits if h.sentence.section_key == key
                      and not self.memory.was_shown(h.sentence)
                      and h.semantic >= self.th["min_relevance"] * 0.8]
            if picked:
                return picked[: config.MAX_SENTENCES_PER_ANSWER]
        return []

    def _recall_text(self) -> str:
        turns = self.memory.content_turns()
        if not turns:
            return "You haven't asked me anything about the documents yet."
        lines = []
        for t in turns:
            sections = ", ".join(t.sections)
            tag = " (repeat)" if t.kind == "repeat" else ""
            lines.append(f"- Turn {t.number}: \"{t.question}\"{tag} → {sections}")
        covered = {s for t in turns for s in t.sections}
        return (f"Here's what you've asked so far ({len(turns)} questions, {len(covered)} sections covered):\n"
                + "\n".join(lines))

    def reset(self) -> None:
        self.memory.reset()
