"""All tunable settings in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"

# Free, local sentence-transformers model (~90 MB, downloaded once, then cached).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid search: final score = ALPHA * semantic + (1 - ALPHA) * keyword (BM25).
HYBRID_ALPHA = 0.7

# Max sentences put into one answer.
MAX_SENTENCES_PER_ANSWER = 3

# Thresholds depend on the embedding backend because their score ranges differ.
#   min_relevance   - best semantic score below this  -> "not in the documents"
#   repeat_question - question this similar to an earlier one -> treated as a repeat
#   weak_query      - a vague question ("what about it?") scoring below this needs context
#   clear_topic     - a question scoring above this on its own clearly names a new topic,
#                     so it is never treated as a follow-up (e.g. "can I recover it?")
THRESHOLDS = {
    "sentence-transformers": {"min_relevance": 0.35, "repeat_question": 0.88, "weak_query": 0.45,
                              "clear_topic": 0.55},
    "tfidf":                 {"min_relevance": 0.12, "repeat_question": 0.75, "weak_query": 0.30,
                              "clear_topic": 0.50},
}
