"""Conversation memory for one chat session."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loader import Sentence


@dataclass
class Turn:
    number: int
    question: str
    kind: str                        # answer | follow_up | repeat | out_of_scope | smalltalk | recall
    vector: np.ndarray | None = None
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def sections(self) -> list[str]:
        """Labels of the sections used in this turn, e.g. ["§2.3 Refund Policy"]."""
        return list(dict.fromkeys(s.label for s in self.sentences))


class SessionMemory:
    """Remembers questions, the current topic, and every sentence already shown."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.shown: dict[int, int] = {}            # sentence id -> turn it was shown in
        self.current: Sentence | None = None       # anchor sentence of the current topic

    @property
    def turn_number(self) -> int:
        return len(self.turns) + 1

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        for s in turn.sentences:
            self.shown.setdefault(s.id, turn.number)
        if turn.sentences:
            self.current = turn.sentences[0]

    def was_shown(self, sentence: Sentence) -> bool:
        return sentence.id in self.shown

    def content_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.sentences]

    def last_content_turn(self) -> Turn | None:
        turns = self.content_turns()
        return turns[-1] if turns else None

    def most_similar_question(self, vector: np.ndarray) -> tuple[Turn | None, float]:
        best, best_sim = None, 0.0
        for t in self.content_turns():
            if t.vector is None or t.kind == "repeat":
                continue
            sim = float(np.dot(t.vector, vector))
            if sim > best_sim:
                best, best_sim = t, sim
        return best, best_sim

    def reset(self) -> None:
        self.__init__()
