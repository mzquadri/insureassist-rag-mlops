"""Score ALCE citation recall and precision over the answers already generated.

`eval/citation_structure.json` records what the generator cited. This asks whether those
citations hold up: does the block say what the sentence claims.

The judge is an entailment model, not the generator, which is ALCE's answer to the
circularity `docs/LIMITATIONS.md` objects to. A different model doing a different task
can be wrong, but it cannot be wrong in the generator's favour by construction.

**The judge is controlled before it is trusted.** A sentence lifted verbatim out of a
block must be entailed by it, and a sentence from an unrelated block must not. Both are
built from the committed corpus, so the control measures the judge on this domain rather
than on MNLI. A low citation score from an untrustworthy judge is unreadable, and the
control is what separates the two.

    python eval/citation_correctness_run.py
    python eval/citation_correctness_run.py --variant ranged

No generation happens here; answers come from the artefact. Needs transformers and a
~370MB NLI model on first run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.citation_correctness import (
    citation_precision,
    citation_recall,
    claims,
)
from src.corpus import chunk_corpus, load_corpus

#: A mid-sized NLI model. ALCE used TRUE (T5-11B), which does not fit here; the agreement
#: figures in their paper therefore do not carry over, and the control below is what this
#: result rests on instead.
MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

#: Entailment is the argmax over the three NLI labels. A probability cut would need
#: calibrating, and nothing here has the data to calibrate it.
REFERENCE = ROOT / "eval" / "reference_run.json"


def artefact_for(variant: str) -> Path:
    suffix = "" if variant == "served" else f"_{variant}"
    return ROOT / "eval" / f"citation_structure{suffix}.json"


def out_for(variant: str) -> Path:
    suffix = "" if variant == "served" else f"_{variant}"
    return ROOT / "eval" / f"citation_correctness{suffix}.json"


def build_judge():
    """Return (entails, describe). Loaded lazily so importing this module stays cheap."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    labels = {i: name.lower() for i, name in model.config.id2label.items()}

    def entails(premise: str, hypothesis: str) -> bool:
        if not premise.strip() or not hypothesis.strip():
            return False
        with torch.no_grad():
            encoded = tokeniser(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            scores = model(**encoded).logits.softmax(-1)[0]
        return labels[int(scores.argmax())] == "entailment"

    return entails, {
        "model": MODEL,
        "device": device,
        "decision": "argmax over NLI labels",
    }


def _well_formed(text: str) -> list[str]:
    """Complete sentences only, for the control.

    Chunks are cut at 800 characters, so they routinely begin and end mid-word: "r
    Coverage D is in addition to ...", "## Appendix A(1) to Part 61". A fragment is not
    a proposition and an entailment model is right to decline it, so a control built
    from fragments measures the chunker rather than the judge. The first attempt at this
    scored 11 of 30 for exactly that reason.
    """
    out = []
    for line in re.split(r"(?<=[.!?])\s+", text):
        candidate = line.strip()
        words = candidate.split()
        if not (8 <= len(words) <= 40):
            continue
        if "#" in candidate or "\n" in candidate:
            continue
        if not candidate[:1].isupper() or candidate[-1] not in ".!?":
            continue
        out.append(candidate)
    return out


def control(blocks_by_question: dict[str, dict[int, str]], entails) -> dict[str, Any]:
    """Does the judge work on this corpus at all?

    Positive: a sentence taken verbatim from a block, which that block must entail.
    Negative: a sentence from a different document, which it must not.
    """
    sentences: list[tuple[str, str]] = []
    for blocks in blocks_by_question.values():
        for text in blocks.values():
            for line in _well_formed(text):
                sentences.append((text, line))
                break

    positives = sentences[:30]
    negatives = [
        (sentences[i][0], sentences[(i + len(sentences) // 2) % len(sentences)][1])
        for i in range(min(30, len(sentences)))
    ]

    hits = sum(1 for premise, hypothesis in positives if entails(premise, hypothesis))
    false_alarms = sum(
        1 for premise, hypothesis in negatives if entails(premise, hypothesis)
    )
    return {
        "positive_pairs": len(positives),
        "positive_entailed": hits,
        "negative_pairs": len(negatives),
        "negative_entailed": false_alarms,
        "note": (
            "Positives are sentences lifted verbatim from the block used as premise, so a "
            "judge that works should entail nearly all of them. Negatives pair a block "
            "with a sentence from elsewhere in the corpus and should be entailed rarely. "
            "Both are drawn from the committed corpus, so this measures the judge on "
            "policy prose rather than on MNLI."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="citation_correctness_run")
    parser.add_argument("--variant", default="served")
    args = parser.parse_args()

    structure = json.loads(artefact_for(args.variant).read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    chunking = reference["retrieval"]["chunking"]
    top_k = int(reference["retrieval"]["serving_top_k"])

    by_id = {
        c.chunk_id: c
        for c in chunk_corpus(load_corpus(), chunking["size"], chunking["overlap"])
    }
    retrieved = {
        q["question_id"]: q["retrieved_chunk_ids"][:top_k]
        for q in reference["per_question"]
    }
    blocks_by_question = {
        qid: {i: by_id[cid].text for i, cid in enumerate(ids, start=1)}
        for qid, ids in retrieved.items()
    }

    entails, judge = build_judge()
    judge_control = control(blocks_by_question, entails)

    rows: list[dict[str, Any]] = []
    all_claims = []
    for row in structure["per_question"]:
        qid = row["question_id"]
        blocks = blocks_by_question[qid]
        items = claims(row["answer"], n_contexts=row["n_contexts"])
        all_claims.extend((c, blocks) for c in items)
        rows.append(
            {
                "question_id": qid,
                "cited_sentences": len(items),
                "recall": citation_recall(items, blocks, entails),
                "precision": citation_precision(items, blocks, entails),
            }
        )
        print(f"  {qid:<10} {len(items)} cited sentence(s)", flush=True)

    # Pooled over sentences rather than averaged over questions: a question with four
    # cited sentences carries four times the evidence of one with a single sentence.
    flat = [c for c, _ in all_claims]
    pooled_blocks: dict[int, str] = {}
    pooled: list[Any] = []
    next_key = 1
    for claim, blocks in all_claims:
        # Every (claim, citation) gets a key no other claim can reach. Deriving keys from
        # a running length collides - [1,3] then [2] then [1,2] all land on 4 - and a
        # collision silently scores one question's sentence against another's block.
        remap: dict[int, int] = {}
        for n in claim.citations:
            remap[n] = next_key
            pooled_blocks[next_key] = blocks.get(n, "")
            next_key += 1
        pooled.append(
            type(claim)(sentence=claim.sentence, citations=list(remap.values()))
        )

    payload = {
        "prompt_variant": args.variant,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metric": "ALCE citation recall and precision (Gao et al., arXiv:2305.14627)",
        "judge": judge,
        "judge_control": judge_control,
        "caveats": (
            "ALCE validated against TRUE (T5-11B) and reported Cohen's kappa 0.698 for "
            "recall; a smaller judge does not inherit that. Correctness is not "
            "faithfulness: a block supporting a claim does not establish the model "
            "derived the claim from it."
        ),
        "summary": {
            "cited_sentences": len(flat),
            "citation_recall": citation_recall(pooled, pooled_blocks, entails),
            "citation_precision": citation_precision(pooled, pooled_blocks, entails),
        },
        "per_question": rows,
    }

    out = out_for(args.variant)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    c = judge_control
    s = payload["summary"]
    print()
    print(
        f"judge control  positives entailed {c['positive_entailed']}/{c['positive_pairs']}"
        f"   negatives entailed {c['negative_entailed']}/{c['negative_pairs']}"
    )
    print(f"cited sentences {s['cited_sentences']}")
    print(f"citation recall    {s['citation_recall']}")
    print(f"citation precision {s['citation_precision']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
