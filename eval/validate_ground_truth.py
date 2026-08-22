"""
Validate the retrieval ground truth against the real corpus.

    python -m eval.validate_ground_truth

Exits non-zero on any broken reference or invalid label. Run it after changing the corpus,
the chunking configuration, or the labels - all three can invalidate each other silently.
"""
from __future__ import annotations

import sys

from eval.ground_truth import load_questions, summarise, validate
from src.config import cfg
from src.corpus import chunk_corpus, load_corpus, verify_document_hashes


def main() -> int:
    documents = load_corpus()

    mismatched = verify_document_hashes(documents)
    if mismatched:
        print("FAIL: corpus files do not match their recorded hashes:", ", ".join(mismatched))
        print("      Chunk IDs derive from this text, so every label is suspect.")
        return 1

    chunks = chunk_corpus(documents, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    questions = load_questions()

    print(f"corpus:    {len(documents)} documents, {len(chunks)} chunks "
          f"(size={cfg.CHUNK_SIZE}, overlap={cfg.CHUNK_OVERLAP})")
    print(f"questions: {len(questions)}")

    problems = validate(questions, documents, chunks)
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    stats = summarise(questions)
    print("\nAll labels resolve against the corpus.\n")
    print(f"  answerable        {stats['answerable']}")
    print(f"  unanswerable      {stats['unanswerable']}  ({stats['unanswerable_share']:.0%})")
    print(f"  by split          {stats['by_split']}")
    print(f"  by difficulty     {stats['by_difficulty']}")
    print("  by category")
    for category, count in stats["by_category"].items():
        print(f"    {category:18} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
