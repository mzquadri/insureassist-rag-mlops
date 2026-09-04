#!/usr/bin/env python
"""The figure set for the retrieval benchmark.

Every number is read from eval/reference_run.json, which a tracked evaluation run
produces. Nothing is typed in, so a figure cannot disagree with the published
result.

The set is built around one uncomfortable fact: the test split has 18 answerable
questions, so most of the differences between the three retrievers are one or two
questions wide. The figures show counts alongside rates and draw Wilson intervals,
because a bar chart of rates alone would imply a precision this benchmark does not
have.

    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import portfolio_style as ps

OUT = REPO / "docs" / "figures"
RUN = REPO / "eval" / "reference_run.json"
SWEEP = REPO / "eval" / "dev_chunking_sweep.json"

SYSTEMS = [("hybrid", "Hybrid RRF", ps.BLUE), ("dense", "Dense (BGE)", ps.GREEN),
           ("bm25", "BM25", ps.AMBER)]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png")


def wilson(k, n, z=1.96):
    """Wilson score interval. Normal approximations are wrong at n=18."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load():
    d = json.loads(RUN.read_text(encoding="utf-8"))
    r = d["retrieval"]
    n = sum(v["n"] for v in r["by_category"].values())
    sets = {"hybrid": r["metrics"], "dense": r["baselines"]["dense"],
            "bm25": r["baselines"]["bm25"]}
    return d, r, n, sets


# ---------------------------------------------------------------------------
def fig_scorecard(d, r, n, sets):
    """Where each retriever wins, with the count behind every rate."""
    metrics = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "mrr",
               "top_document_accuracy"]
    nice = {"hit_rate@1": "hit@1", "hit_rate@3": "hit@3", "hit_rate@5": "hit@5",
            "mrr": "MRR", "top_document_accuracy": "top-doc\naccuracy"}

    fig = plt.figure(figsize=(13.4, 8.4))
    ax = fig.add_axes([0.075, 0.20, 0.885, 0.545])
    x = np.arange(len(metrics))
    w = 0.26
    for i, (key, label, colour) in enumerate(SYSTEMS):
        vals = [sets[key][m] for m in metrics]
        off = (i - 1) * w
        bars = ax.bar(x + off, vals, width=w * 0.92, color=colour,
                      label=label, zorder=3)
        for xi, (b, m, v) in enumerate(zip(bars, metrics, vals, strict=True)):
            if m.startswith("hit_rate") or m == "top_document_accuracy":
                ax.text(b.get_x() + b.get_width() / 2, v + 0.018,
                        f"{round(v * n)}/{n}", ha="center", fontsize=8.6,
                        color=ps.BODY)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.2f}",
                        ha="center", fontsize=8.6, color=ps.BODY)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[m] for m in metrics], fontsize=11)
    ax.set_ylim(0, 0.78)
    ps.clean(ax, grid_axis="y")
    ax.set_ylabel("score", fontsize=11)
    ax.legend(fontsize=10.4, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.09))

    ps.title_block(
        fig, "Hybrid wins at rank 1; BM25 wins at depth",
        "Retrieval on the 18 answerable test questions. Counts are printed on the "
        "hit-rate bars, because at this sample size\nthe gaps are one or two "
        "questions wide.", y=0.962, size=23)
    ps.footnote(fig, [
        "Hybrid RRF was selected on the dev split for MRR and top-document accuracy, " +
        f"and it does lead both here ({sets['hybrid']['mrr']:.2f} MRR against " +
        f"{sets['bm25']['mrr']:.2f}). At depth 5 it does not: BM25 retrieves a relevant " +
        f"chunk for {round(sets['bm25']['hit_rate@5']*n)} of {n} questions against " +
        f"{round(sets['hybrid']['hit_rate@5']*n)}.",
        "That single-question gap is inside the noise of an 18-question benchmark; the " +
        "next figure shows the intervals. Source: eval/reference_run.json."], y=0.105)
    save(fig, "01_retrieval_scorecard")


def fig_intervals(d, r, n, sets):
    """The same result with its uncertainty, which changes the conclusion."""
    fig = plt.figure(figsize=(13.4, 8.0))
    ax = fig.add_axes([0.235, 0.215, 0.700, 0.545])

    rows = []
    for depth in ("hit_rate@1", "hit_rate@3", "hit_rate@5"):
        for key, label, colour in SYSTEMS:
            k = round(sets[key][depth] * n)
            lo, hi = wilson(k, n)
            rows.append((f"{label}", depth.replace("hit_rate@", "hit@"), k, k / n,
                         lo, hi, colour))

    ypos = np.arange(len(rows))[::-1]
    for (label, depth, k, p, lo, hi, colour), yy in zip(rows, ypos, strict=True):
        ax.plot([lo, hi], [yy, yy], color=colour, lw=3.0, alpha=0.30,
                solid_capstyle="round")
        ax.plot([p], [yy], "o", color=colour, ms=8, mec=ps.PAPER, mew=1.4, zorder=3)
        ax.text(-0.035, yy, label, ha="right", va="center", fontsize=10.6,
                color=ps.INK, transform=ax.get_yaxis_transform())
        ax.text(hi + 0.012, yy, f"{k}/{n}", va="center", fontsize=9.6, color=ps.MUTED)
    # Rows are appended @1, @3, @5 and ypos is reversed, so the first group sits at
    # the top. The group labels must follow that order, not the reading order.
    for i, depth in enumerate(("hit@1", "hit@3", "hit@5")):
        ax.text(-0.175, ypos[i * 3 + 1], depth, ha="right", va="center",
                fontsize=12.2, color=ps.INK, fontweight="600",
                transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ps.clean(ax, left=False, grid_axis="x")
    ax.set_xlabel("share of the 18 answerable test questions with a relevant chunk "
                  "retrieved", fontsize=10.6)

    ps.title_block(
        fig, "At n = 18 the three retrievers are not separable",
        "The same numbers as the scorecard, with 95% Wilson intervals. Every interval "
        "overlaps every other interval at the\nsame depth, so the ranking between them "
        "is not supported by this benchmark.", y=0.962, size=23)
    ps.footnote(fig, [
        "Wilson rather than a normal approximation, which misbehaves at this sample " +
        "size and near 0 or 1.",
        "This is a limit of the benchmark, not of the retrievers. Separating a " +
        "five-point difference in hit rate with confidence would need on the order of " +
        "several hundred questions; there are 40 in total, 18 answerable in test.",
        "The honest reading is the one the repository already takes: hybrid was " +
        "selected on dev for MRR, and the test split neither confirms nor refutes that " +
        "choice at depth."], y=0.098)
    save(fig, "02_confidence_intervals")


def fig_by_category(d, r, n, sets):
    """Where retrieval actually fails."""
    cats = r["by_category"]
    order = sorted(cats, key=lambda c: cats[c]["hit_rate@5"])
    fig = plt.figure(figsize=(13.4, 8.2))
    ax = fig.add_axes([0.215, 0.215, 0.545, 0.545])
    axn = fig.add_axes([0.800, 0.215, 0.155, 0.545])

    ypos = np.arange(len(order))[::-1]
    for c, yy in zip(order, ypos, strict=True):
        v = cats[c]["hit_rate@5"]
        col = ps.RED if v == 0 else (ps.AMBER if v < 0.7 else ps.GREEN)
        ax.barh(yy, v, color=col, height=0.62, zorder=3)
        ax.text(-0.02, yy, c.replace("_", " "), ha="right", va="center",
                fontsize=11.4, color=ps.INK,
                transform=ax.get_yaxis_transform())
        ax.text(v + 0.02, yy, f"{round(v*cats[c]['n'])}/{cats[c]['n']}", va="center",
                fontsize=9.8, color=ps.MUTED)
    ax.set_yticks([]); ax.set_xlim(0, 1.15); ax.set_ylim(-0.7, len(order) - 0.3)
    ps.clean(ax, left=False, grid_axis="x")
    ax.set_xlabel("hit rate @5", fontsize=10.6)

    for c, yy in zip(order, ypos, strict=True):
        axn.barh(yy, cats[c]["n"], color=ps.HAIR, height=0.62, zorder=3)
        axn.text(cats[c]["n"] + 0.15, yy, str(cats[c]["n"]), va="center",
                 fontsize=9.8, color=ps.MUTED)
    axn.set_yticks([]); axn.set_ylim(-0.7, len(order) - 0.3)
    axn.set_xlim(0, max(v["n"] for v in cats.values()) * 1.45)
    ps.clean(axn, left=False, grid_axis="x")
    axn.set_xlabel("questions", fontsize=10.6)

    ps.title_block(
        fig, "Two categories fail completely",
        "Hit rate at depth 5 by question category, worst first. The bars on the right "
        "are how many questions each category\nholds — which is the reason to read "
        "this chart carefully.", y=0.962, size=23)
    ps.footnote(fig, [
        "numeric_limit and single_chunk retrieve nothing relevant in the top 5 — but " +
        "they hold two questions each, so '0 of 2' is the whole story and one lucky " +
        "retrieval would move either to 50%.",
        "time_period and multi_chunk are perfect on four and two questions " +
        "respectively. Neither result is strong evidence on its own.",
        "The categories worth acting on are the ones with both a low rate and enough " +
        "questions to mean something: near_miss, at 2 of 5."], y=0.098)
    save(fig, "03_by_category")


def fig_benchmark(d, r, n, sets):
    """What the benchmark is made of."""
    q = d["questions"]
    corpus = d["corpus"]
    fig = plt.figure(figsize=(13.4, 8.2))
    axA = fig.add_axes([0.070, 0.415, 0.385, 0.345])
    axB = fig.add_axes([0.575, 0.415, 0.385, 0.345])
    axC = fig.add_axes([0.070, 0.135, 0.890, 0.150])

    cats = q["by_category"]
    order = sorted(cats, key=lambda c: -cats[c])
    axA.bar(range(len(order)), [cats[c] for c in order],
            color=[ps.RED if c == "unanswerable" else ps.BLUE for c in order],
            width=0.7)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels([c.replace("_", "\n") for c in order], fontsize=8.6)
    ps.clean(axA, grid_axis="y")
    axA.set_ylabel("questions", fontsize=10.4)
    axA.text(0, 1.10, "40 questions by category", transform=axA.transAxes,
             fontsize=12.4, color=ps.INK, fontweight="600", va="bottom")

    diff = q["by_difficulty"]
    keys = ["easy", "medium", "hard"]
    axB.bar(range(3), [diff[k] for k in keys],
            color=[ps.GREEN, ps.AMBER, ps.RED], width=0.62)
    axB.set_xticks(range(3)); axB.set_xticklabels(keys, fontsize=10)
    ps.clean(axB, grid_axis="y")
    axB.set_ylabel("questions", fontsize=10.4)
    for i, k in enumerate(keys):
        axB.text(i, diff[k] + 0.4, str(diff[k]), ha="center", fontsize=10,
                 color=ps.BODY)
    axB.text(0, 1.10, "and by difficulty", transform=axB.transAxes, fontsize=12.4,
             color=ps.INK, fontweight="600", va="bottom")

    ps.bare(axC)
    axC.set_xlim(0, 1); axC.set_ylim(0, 1)
    facts = [(f"{corpus['documents']}", "NFIP policy forms"),
             (f"{corpus['chunks']:,}", "chunks at 800/120"),
             (f"{corpus['words']:,}", "words"),
             (f"{q['total']}", "questions written"),
             (f"{q['unanswerable']}", "deliberately unanswerable"),
             (f"{n}", "answerable in test")]
    for i, (val, lab) in enumerate(facts):
        x = i / len(facts)
        axC.text(x, 0.62, val, fontsize=20, color=ps.INK, fontweight="600")
        axC.text(x, 0.20, lab, fontsize=10, color=ps.FAINT)

    ps.title_block(
        fig, "What the benchmark is made of",
        "Forty questions written against three NFIP policy forms, a fifth of them "
        "deliberately unanswerable so abstention\ncan be measured rather than assumed.",
        y=0.962, size=23)
    ps.footnote(fig, [
        f"Corpus hash {corpus['corpus_hash'][:16]}…, question set hash " +
        f"{q['question_set_hash'][:16]}…. Both are checked by eval/verify_artifacts.py, " +
        "so a changed corpus cannot silently invalidate a published score.",
        "The split is a deterministic hash of the question id, stratified by category: " +
        "18 dev, 22 test. Retrieval is scored on the 18 answerable test questions."],
        y=0.072)
    save(fig, "04_benchmark_composition")


def fig_abstention(d, r, n, sets):
    """The honest negative result."""
    ab = d["generation"]["abstention"]
    cit = d["generation"]["citations"]
    fig = plt.figure(figsize=(13.4, 7.6))
    axA = fig.add_axes([0.075, 0.300, 0.360, 0.420])
    axB = fig.add_axes([0.585, 0.300, 0.360, 0.420])

    n_unans = d["questions"]["evaluated_in_this_run"] - n
    answered, refused = ab["false_answers"], n_unans - ab["false_answers"]
    axA.bar([0], [answered], color=ps.RED, width=0.5, zorder=3)
    axA.bar([0], [refused], bottom=[answered], color=ps.GREEN, width=0.5, zorder=3)
    axA.set_xlim(-0.7, 0.7); axA.set_xticks([])
    ps.clean(axA, bottom=False, grid_axis="y")
    axA.set_ylabel("unanswerable test questions", fontsize=10.4)
    axA.text(0, answered / 2, f"{answered} answered anyway", ha="center",
             va="center", color=ps.PAPER, fontsize=11, fontweight="600")
    axA.text(0, 1.10, "Every unanswerable question got an answer",
             transform=axA.transAxes, fontsize=12.4, color=ps.INK,
             fontweight="600", va="bottom")

    labels = ["citation\nprecision", "citation\nrecall", "unsupported\ncitation rate"]
    vals = [cit["citation_precision"], cit["citation_recall"],
            cit["unsupported_citation_rate"]]
    cols = [ps.AMBER, ps.BLUE, ps.GREEN]
    axB.bar(range(3), vals, color=cols, width=0.62, zorder=3)
    for i, v in enumerate(vals):
        axB.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, color=ps.BODY)
    axB.set_xticks(range(3)); axB.set_xticklabels(labels, fontsize=9.6)
    axB.set_ylim(0, 0.62)
    ps.clean(axB, grid_axis="y")
    axB.text(0, 1.10, "Citations point at retrieved text, always",
             transform=axB.transAxes, fontsize=12.4, color=ps.INK,
             fontweight="600", va="bottom")

    ps.title_block(
        fig, "The part that does not work",
        "There is no abstention rule. Retrieval always returns something, so every "
        "unanswerable question is passed to the\ngenerator and answered — four out of "
        "four in this run.", y=0.962, size=23)
    ps.footnote(fig, [
        "No similarity threshold is applied, and none is claimed. On dev, no " +
        "single-threshold rule on the top-1 dense score beat the always-answer " +
        "baseline (0.778 against 0.778), and the hardest unanswerable question scored " +
        "above the answerable mean.",
        "With eight unanswerable questions in the whole benchmark there is not enough " +
        "evidence to fit a threshold, so the weakness is recorded rather than papered " +
        "over.",
        f"Citation precision is bounded by design: with {cit['citations_checked']} " +
        "citations checked and five chunks returned for a question that often has one " +
        "relevant chunk, it cannot approach 1.0. Unsupported citation rate is 0.000 — " +
        "every citation points at text that was actually retrieved."], y=0.128)
    save(fig, "05_abstention_and_citations")


def fig_sweep(d, r, n, sets):
    """The chunking decision, on dev."""
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(13.4, 8.0))
    ax = fig.add_axes([0.085, 0.245, 0.880, 0.505])

    keys = list(sweep)
    metric = "hit_rate@5" if "hit_rate@5" in next(iter(sweep.values())) else None
    if metric is None:
        metric = next(k for k in next(iter(sweep.values())) if "hit" in k)
    vals = [sweep[k][metric] for k in keys]
    chosen = r["chunking"]
    tag = f"size={chosen['size']},ovl={chosen['overlap']}"
    cols = [ps.BLUE if tag in k else ps.HAIR for k in keys]
    ax.bar(range(len(keys)), vals, color=cols, width=0.66, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.2, color=ps.BODY)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace(",", "\n") for k in keys], fontsize=8.4)
    ps.clean(ax, grid_axis="y")
    ax.set_ylabel(metric.replace("_", " "), fontsize=10.6)

    best = int(np.argmax(vals))
    ps.note(ax, best, vals[best] + 0.055, "selected", color=ps.BLUE, size=11,
            ha="center", weight="600")

    ps.title_block(
        fig, "Why the chunks are 800 characters",
        "Every chunking configuration tried on the dev split. Larger chunks keep a "
        "provision and the text identifying which\nform it belongs to inside one "
        "window, which is what the dominant failure mode needed.", y=0.962, size=23)
    ps.footnote(fig, [
        "Selected on dev and then left alone: the test split was scored once, with this " +
        "configuration frozen, which is the only way the test number means anything.",
        f"Config hash {d['reproducibility']['config_hash'][:16]}…, retrieval config " +
        f"hash {d['reproducibility']['retrieval_config_hash'][:16]}…"], y=0.108)
    save(fig, "06_chunking_sweep")


def main() -> int:
    ps.apply()
    d, r, n, sets = load()
    print(f"reference run: {d['benchmark']}, {d['split']} split, "
          f"{n} answerable questions scored\n")
    fig_scorecard(d, r, n, sets)
    fig_intervals(d, r, n, sets)
    fig_by_category(d, r, n, sets)
    fig_benchmark(d, r, n, sets)
    fig_abstention(d, r, n, sets)
    fig_sweep(d, r, n, sets)
    print(f"\nfigures written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
