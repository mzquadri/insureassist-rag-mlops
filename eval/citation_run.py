"""Generate answers over the pinned retrieval and measure what they did with citations.

`docs/LIMITATIONS.md` says two things about generation. No answer-quality metric is
published, because the only local model grading its own output would be circular. And
the model is prompted to cite context numbers, but whether it does so correctly is not
measured.

The second has a measurable half that the first does not reach. Whether an answer cited
at all, and whether the blocks it cited were ones it was actually given, are facts about
the string. No judge is involved, so there is nothing to be circular about.

Retrieval is held fixed. The contexts come from `retrieved_chunk_ids` in the pinned
reference run rather than from a live query, so this measures the generator against
exactly the retrieval the published numbers describe, and needs no Qdrant.

Generation is pinned to temperature 0 with a fixed seed. **That is not how the service
runs** - serving uses the backend default and is non-deterministic - and the difference
is declared in the artifact rather than left for a reader to assume.

    python eval/citation_run.py          # needs Ollama; writes eval/citation_structure.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import urllib.error
import urllib.request

from eval.citation_structure import summarise, use_of_citations
from src.config import cfg
from src.corpus import chunk_corpus, load_corpus
from src.rag import build_prompt

REFERENCE = ROOT / "eval" / "reference_run.json"
OUT = ROOT / "eval" / "citation_structure.json"

#: Pinned so a rerun gives the same answers. Serving does not do this; see the docstring.
TEMPERATURE = 0.0
SEED = 17
TIMEOUT = 180


class GenerationUnavailable(RuntimeError):
    """Ollama is not reachable, or refused."""


def generate(prompt: str, model: str, url: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "seed": SEED},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return str(json.loads(response.read())["response"])
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        KeyError,
    ) as exc:
        raise GenerationUnavailable(f"{model} at {url} did not answer: {exc}") from exc


def contexts_for(
    chunk_ids: list[str], by_id: dict, titles: dict[str, Any]
) -> list[dict]:
    """Rebuild the context blocks the prompt builder expects, in retrieved order."""
    blocks = []
    for chunk_id in chunk_ids:
        chunk = by_id[chunk_id]
        document = titles[chunk.document_id]
        blocks.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": document.title,
                "cfr_citation": document.cfr_citation,
            }
        )
    return blocks


def main() -> int:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    chunking = reference["retrieval"]["chunking"]
    top_k = int(reference["retrieval"]["serving_top_k"])

    documents = load_corpus()
    by_document = {d.document_id: d for d in documents}
    by_id = {
        c.chunk_id: c
        for c in chunk_corpus(documents, chunking["size"], chunking["overlap"])
    }

    questions = {}
    for line in (
        (ROOT / "eval" / "ground_truth" / "nfip_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line.strip():
            row = json.loads(line)
            questions[row["question_id"]] = row

    model = cfg.OLLAMA_MODEL
    url = cfg.OLLAMA_URL

    # Answers are cached per question. Generation on a local 3B model is minutes per
    # question, so a run that lost everything to one timeout would be unfinishable on
    # the hardware this is meant to run on. The cache is keyed by the prompt, so a
    # changed prompt or a changed retrieval regenerates rather than reusing silently.
    cache_path = ROOT / "eval" / ".citation_answers.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    uses = []
    for entry in reference["per_question"]:
        question_id = entry["question_id"]
        question = questions[question_id]["question"]
        chunk_ids = entry["retrieved_chunk_ids"][:top_k]
        blocks = contexts_for(chunk_ids, by_id, by_document)

        prompt = build_prompt(question, blocks)
        key = hashlib.sha256(
            f"{model}\n{TEMPERATURE}\n{SEED}\n{prompt}".encode()
        ).hexdigest()
        if key in cache:
            answer = cache[key]
        else:
            answer = generate(prompt, model, url)
            cache[key] = answer
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        used = use_of_citations(answer, n_contexts=len(blocks))
        uses.append(used)
        rows.append(
            {
                "question_id": question_id,
                "category": entry.get("category"),
                "n_contexts": len(blocks),
                "answer_chars": len(answer),
                # The answer is kept so the parsing above can be checked rather than
                # trusted. A citation metric nobody can audit is the same shape of
                # claim this repository objects to elsewhere.
                "answer": answer,
                **used.as_dict(),
            }
        )
        flag = "" if used.cited_any and not used.out_of_range else "   <-"
        # Flushed: on a local model each question is minutes, and a buffered
        # progress line is indistinguishable from a hang.
        print(
            f"  {question_id:<10} cited {used.cited!s:<16} of {len(blocks)}{flag}",
            flush=True,
        )

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what_this_measures": (
            "Whether the generator cited at all, and whether the blocks it cited were "
            "supplied to it. Facts about the string; no judge, so no circularity. It "
            "does not measure whether a cited block supports the sentence, which is "
            "citation correctness and needs an entailment model."
        ),
        "generation": {
            "backend": "ollama",
            "model": model,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "differs_from_serving": (
                "Serving uses the backend default and is non-deterministic. These "
                "settings are pinned so the run reproduces, and are not what the "
                "service does."
            ),
        },
        "retrieval": {
            "source": "eval/reference_run.json per_question",
            "note": "Held fixed at the published retrieval; this measures generation only.",
            "serving_top_k": top_k,
            "chunking": chunking,
        },
        "summary": summarise(uses),
        "per_question": rows,
    }

    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    s = payload["summary"]
    print()
    print(f"answers                 {s['answers']}")
    print(f"cited something         {s['cited_any']}  ({1 - s['uncited_rate']:.1%})")
    print(f"cited nothing           {s['uncited']}  ({s['uncited_rate']:.1%})")
    print(
        f"cited a block not given {s['with_out_of_range']}  ({s['out_of_range_rate']:.1%})"
    )
    print(f"mean context coverage   {s['mean_context_coverage']:.1%} of {top_k} blocks")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationUnavailable as exc:
        print(f"generation unavailable: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
