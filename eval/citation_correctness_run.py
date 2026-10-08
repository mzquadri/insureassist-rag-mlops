"""Score ALCE citation recall and precision over the answers already generated.

`eval/citation_structure.json` records what the generator cited. This asks whether those
citations hold up: does the block say what the sentence claims.

The judge is an entailment model, not the generator, which is ALCE's answer to the
circularity `docs/LIMITATIONS.md` objects to. A different model doing a different task
can be wrong, but it cannot be wrong in the generator's favour by construction.

**The judge is calibrated before it is trusted.** A low citation score from an
untrustworthy judge is unreadable, and the calibration is what separates the two. It is
built from the ground truth: a human-written `gold_answer` against the chunk holding its
labelled evidence offsets must be entailed, and the same answer against a chunk the
labels do not mark relevant must not. That is the shape of a citation - a paraphrase
supported by a passage - which is why it sets the threshold.

A second control pairs each block with a sentence copied verbatim out of it. It is still
run and recorded, as a check that the judge works at all, but it does **not** set the
threshold: copies score near 1.0, so calibrating on them puts the cut near 1.0 and
rejects the supported paraphrases the metric exists to count. A control has to be
representative as well as discriminative.

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
    citation_precision_counts,
    citation_recall,
    citation_recall_counts,
    claims,
)
from eval.nli_aggregation import (
    batch_plan,
    best_scores,
    best_sentence_score,
    paraphrase_control,
    rates,
    youden_threshold,
)
from src.corpus import chunk_corpus, load_corpus

#: Candidate judges. ALCE used TRUE (T5-11B), which does not fit on a 2 GiB card, so the
#: agreement figures in their paper do not carry over and the calibration below is what any
#: result rests on instead. The base model clears it once the premise is split and the cut
#: is calibrated on paraphrase; the larger one was tried while the base model was still
#: failing, and over-entails badly enough that it is kept only as a recorded negative.
JUDGES = {
    "base": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}
DEFAULT_JUDGE = "base"

#: Padded tokens per forward pass. The card this runs on holds 2 GiB, about 1.6 GiB of
#: it free, and DeBERTa's disentangled attention builds several batch x len x len tensors
#: per layer. 4,096 is a batch of 8 at full length or 400 short sentence pairs; `forward`
#: halves anything that still will not fit.
TOKEN_BUDGET = 4096
MAX_TOKENS = 512

#: Declared before the run, following eval/abstention_threshold.py, which fixes its
#: sensitivity floor in the script rather than choosing one after seeing the result. If
#: the calibrated judge misses either floor on the control, no rate is reported.
MIN_SENSITIVITY = 0.80
MIN_SPECIFICITY = 0.80

REFERENCE = ROOT / "eval" / "reference_run.json"


def artefact_for(variant: str) -> Path:
    suffix = "" if variant == "served" else f"_{variant}"
    return ROOT / "eval" / f"citation_structure{suffix}.json"


def out_for(variant: str, judge: str = DEFAULT_JUDGE) -> Path:
    """One file per (arm, judge). A judge that clears the control and one that does not
    are different measurements and must not overwrite each other."""
    parts = "" if variant == "served" else f"_{variant}"
    parts += "" if judge == DEFAULT_JUDGE else f"_{judge}"
    return ROOT / "eval" / f"citation_correctness{parts}.json"


def build_judge(name: str = DEFAULT_JUDGE):
    """Return (entails, describe). Loaded lazily so importing this module stays cheap."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = JUDGES[name]
    tokeniser = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    labels = {i: name.lower() for i, name in model.config.id2label.items()}

    entail_index = next(i for i, n in labels.items() if n == "entailment")

    def forward(indices: list[int], pairs: list) -> list[tuple[int, float]]:
        """Score one batch, halving it if the card cannot hold it.

        `TOKEN_BUDGET` is an estimate of what fits, and an estimate that is wrong one way
        loses the whole run. Splitting on OOM makes a bad estimate cost time instead.
        """
        try:
            encoded = tokeniser(
                [pairs[i][0] for i in indices],
                [pairs[i][1] for i in indices],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS,
                padding=True,
            ).to(device)
            with torch.no_grad():
                probabilities = model(**encoded).logits.softmax(-1)
            return [
                (i, float(row[entail_index]))
                for i, row in zip(indices, probabilities, strict=True)
            ]
        except torch.cuda.OutOfMemoryError:
            if len(indices) == 1:
                raise
            torch.cuda.empty_cache()
            middle = len(indices) // 2
            return forward(indices[:middle], pairs) + forward(indices[middle:], pairs)

    def score_many(pairs: list) -> list[float]:
        """Every pair scored in length-sorted, token-budgeted batches.

        One forward pass per sentence does not finish on this hardware - about 1,300 of
        them for one judge - and a fixed batch count runs out of memory, because padding
        is set by the longest member. Lengths are measured once without padding, and
        `batch_plan` groups from there.
        """
        if not pairs:
            return []
        lengths = [
            len(
                tokeniser(premise, hypothesis, truncation=True, max_length=MAX_TOKENS)[
                    "input_ids"
                ]
            )
            for premise, hypothesis in pairs
        ]
        out: list[float] = [0.0] * len(pairs)
        for batch in batch_plan(lengths, TOKEN_BUDGET):
            for index, value in forward(batch, pairs):
                out[index] = value
        return out

    def score(premise: str, hypothesis: str) -> float:
        """Probability of entailment, not a label.

        An argmax applies whatever cut the training mix left behind; the threshold for
        this corpus is chosen on the control instead.
        """
        if not premise.strip() or not hypothesis.strip():
            return 0.0
        with torch.no_grad():
            encoded = tokeniser(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(device)
            probabilities = model(**encoded).logits.softmax(-1)[0]
        return float(probabilities[entail_index])

    return (score, score_many), {
        "name": name,
        "model": checkpoint,
        "device": device,
        "decision": "entailment probability, max over premise sentences, calibrated cut",
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


def control(
    blocks_by_question: dict[str, dict[int, str]], score_many
) -> dict[str, Any]:
    """Calibrate the judge on this corpus, and say whether it is usable.

    Positive: a sentence taken verbatim from a block, which that block must entail.
    Negative: a sentence from elsewhere in the corpus, which it must not.

    The cut is chosen here, on corpus sentences, and applied to citation pairs, which are
    a different set. Fitting it on the pairs being measured would be scoring the system
    against a threshold tuned to flatter it.
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

    positive_scores = best_scores(positives, score_many)
    negative_scores = best_scores(negatives, score_many)
    threshold = youden_threshold(positive_scores, negative_scores)

    measured = rates(positive_scores, negative_scores, threshold) if threshold else {}
    usable = bool(
        threshold is not None
        and measured.get("sensitivity", 0) >= MIN_SENSITIVITY
        and measured.get("specificity", 0) >= MIN_SPECIFICITY
    )
    return {
        "positive_pairs": len(positives),
        "positive_entailed": sum(
            1 for s in positive_scores if threshold is not None and s >= threshold
        ),
        "negative_pairs": len(negatives),
        "negative_entailed": sum(
            1 for s in negative_scores if threshold is not None and s >= threshold
        ),
        "threshold": round(threshold, 4) if threshold is not None else None,
        "chosen_by": "Youden's J on the control, as in eval/abstention_threshold.py",
        **measured,
        "declared_floor": {
            "sensitivity": MIN_SENSITIVITY,
            "specificity": MIN_SPECIFICITY,
            "declared": "before the run, in the script",
        },
        "usable": usable,
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
    parser.add_argument("--judge", choices=sorted(JUDGES), default=DEFAULT_JUDGE)
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

    (score, score_many), judge = build_judge(args.judge)

    # Two controls. The verbatim one shows the judge can see literal support and is
    # reported; it is not used to calibrate, because copies score near 1.0 and the cut
    # it chooses is far too strict for paraphrase. The gold answers are paraphrases
    # supported by labelled chunks, which is the shape of the thing being measured, so
    # the cut comes from those.
    rows_gt = [
        json.loads(line)
        for line in (ROOT / "eval" / "ground_truth" / "nfip_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    chunk_texts = {cid: c.text for cid, c in by_id.items()}
    all_chunks = list(by_id.values())
    pos_pairs, neg_pairs = paraphrase_control(rows_gt, chunk_texts, all_chunks)
    pos_scores = best_scores(pos_pairs, score_many)
    neg_scores = best_scores(neg_pairs, score_many)
    cut = youden_threshold(pos_scores, neg_scores)
    calibration = {
        "built_from": "gold_answer against labelled relevant / irrelevant chunks",
        "why": (
            "A citation metric scores paraphrase, not copies. Calibrating on verbatim "
            "text puts the cut near 1.0 and rejects supported paraphrases."
        ),
        "positive_pairs": len(pos_pairs),
        "negative_pairs": len(neg_pairs),
        "threshold": round(cut, 4) if cut is not None else None,
        "chosen_by": "Youden's J, as in eval/abstention_threshold.py",
        **(rates(pos_scores, neg_scores, cut) if cut is not None else {}),
    }
    calibration["usable"] = bool(
        cut is not None
        and calibration.get("sensitivity", 0) >= MIN_SENSITIVITY
        and calibration.get("specificity", 0) >= MIN_SPECIFICITY
    )
    calibration["declared_floor"] = {
        "sensitivity": MIN_SENSITIVITY,
        "specificity": MIN_SPECIFICITY,
        "declared": "before the run, in the script",
    }

    judge_control = control(blocks_by_question, score_many)

    # The calibrated decision the rest of the run uses. Sentence-wise over the premise,
    # then the cut the control chose.

    def entails(premise: str, hypothesis: str) -> bool:
        return (
            cut is not None and best_sentence_score(premise, hypothesis, score) >= cut
        )

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
        "calibration": calibration,
        "judge_control_verbatim": judge_control,
        "caveats": (
            "ALCE validated against TRUE (T5-11B) and reported Cohen's kappa 0.698 for "
            "recall; a smaller judge does not inherit that. Correctness is not "
            "faithfulness: a block supporting a claim does not establish the model "
            "derived the claim from it."
        ),
        # Rates are computed either way so the artefact records what was seen, and are
        # marked unusable when the judge missed its declared floor. The documentation
        # quotes the control, never these, unless `usable` is true.
        # Counts as well as rates. Sixteen cited sentences carry a wide interval, and a
        # reader cannot compute one from a rate alone; precision's denominator is the
        # citations on supported sentences, which is not recoverable from the rate at all.
        "summary": {
            "cited_sentences": len(flat),
            "citation_recall": citation_recall(pooled, pooled_blocks, entails),
            "recall_counts": dict(
                zip(
                    ("supported", "cited"),
                    citation_recall_counts(pooled, pooled_blocks, entails),
                    strict=True,
                )
            ),
            "citation_precision": citation_precision(pooled, pooled_blocks, entails),
            "precision_counts": dict(
                zip(
                    ("needed", "offered"),
                    citation_precision_counts(pooled, pooled_blocks, entails),
                    strict=True,
                )
            ),
            "reportable": calibration["usable"],
        },
        "per_question": rows,
    }

    out = out_for(args.variant, args.judge)
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
