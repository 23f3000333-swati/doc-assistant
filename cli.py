"""Chat with the assistant in the terminal.  Commands: /history  /reset  /quit"""

import argparse
import logging

from doc_assistant import SupportAssistant
from doc_assistant.config import KNOWLEDGE_BASE_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=str(KNOWLEDGE_BASE_DIR),
                        help="folder with your documents (.pdf .docx .txt .md)")
    parser.add_argument("--backend", default="auto", choices=["auto", "sentence-transformers", "tfidf"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    bot = SupportAssistant(docs_dir=args.docs, backend=args.backend)
    print(f"Document support assistant · read {len(bot.kb.documents)} document(s) · backend: {bot.backend}")
    print("Commands: /history  /reset  /quit\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            bot.reset()
            print("Memory cleared.\n")
            continue
        if q == "/history":
            q = "What have I asked so far?"
        print(f"\nAssistant: {bot.ask(q).text}\n")


if __name__ == "__main__":
    main()
