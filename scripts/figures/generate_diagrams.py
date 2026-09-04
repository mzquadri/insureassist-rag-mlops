"""The architecture and lifecycle diagrams, generated from the evidence.

Both diagrams used to describe a system that no longer exists: a Phi-3 model
fine-tuned with LoRA, tracked in MLflow, graded by an LLM-as-judge. None of that
is true of what ships. The generator is llama3.2:3b and is not fine-tuned, no
judge metrics are published, MLflow appears only in the archive, and the retrieval
architecture the whole project is about -- BM25 fused with dense by reciprocal
rank fusion -- was missing from the picture entirely.

So the diagrams are built from eval/reference_run.json rather than drawn by hand,
and the numbers and model names in them come from the run that produced the
published metrics.

    python scripts/figures/generate_diagrams.py

Output: docs/architecture.svg, docs/mlops-pipeline.svg
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "docs"
RUN = REPO / "eval" / "reference_run.json"

INK, MUTED, FAINT, HAIR = "#111827", "#6B7280", "#9CA3AF", "#E5E7EB"
BLUE, BLUE_BG = "#2563EB", "#EFF6FF"
GREEN, GREEN_BG = "#059669", "#ECFDF5"
AMBER, AMBER_BG = "#D97706", "#FFFBEB"
GREY_BG = "#F9FAFB"
FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"


def header(w, h, title, subtitle):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="{FONT}">\n'
        f'  <rect width="{w}" height="{h}" fill="#FFFFFF"/>\n'
        f"  <defs><style>\n"
        f"    .h {{ fill:{INK}; font-size:19px; font-weight:600; }}\n"
        f"    .s {{ fill:{MUTED}; font-size:12.5px; }}\n"
        f"    .lbl {{ fill:{INK}; font-size:13.5px; font-weight:600; }}\n"
        f"    .sub {{ fill:{MUTED}; font-size:11px; }}\n"
        f"    .mono {{ font-family:ui-monospace,Consolas,monospace; font-size:11px; "
        f"fill:#374151; }}\n"
        f"    .cap {{ fill:{FAINT}; font-size:11.5px; }}\n"
        f"    .lane {{ fill:{FAINT}; font-size:11.5px; font-style:italic; }}\n"
        f"  </style></defs>\n"
        f'  <text class="h" x="30" y="36">{title}</text>\n'
        f'  <text class="s" x="30" y="58">{subtitle}</text>\n')


def box(x, y, w, h, fill, edge, title, lines, mono=None):
    s = (f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
         f'stroke="{edge}" stroke-width="1.4"/>\n'
         f'  <text class="lbl" x="{x + w/2}" y="{y + 24}" text-anchor="middle">'
         f'{title}</text>\n')
    for i, ln in enumerate(lines):
        s += (f'  <text class="sub" x="{x + w/2}" y="{y + 43 + i*16}" '
              f'text-anchor="middle">{ln}</text>\n')
    if mono:
        s += (f'  <text class="mono" x="{x + w/2}" y="{y + h - 11}" '
              f'text-anchor="middle">{mono}</text>\n')
    return s


def arrow(x1, y1, x2, y2, label=None):
    head = 7.0
    dx, dy = x2 - x1, y2 - y1
    ln = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / ln, dy / ln
    ex, ey = x2 - ux * head, y2 - uy * head
    px, py = -uy, ux
    s = (f'  <line x1="{x1}" y1="{y1}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{MUTED}" '
         f'stroke-width="1.6"/>\n'
         f'  <polygon points="{x2},{y2} {ex + px*4.4:.1f},{ey + py*4.4:.1f} '
         f'{ex - px*4.4:.1f},{ey - py*4.4:.1f}" fill="{MUTED}"/>\n')
    if label:
        s += (f'  <text class="cap" x="{(x1+x2)/2}" y="{(y1+y2)/2 - 8}" '
              f'text-anchor="middle">{label}</text>\n')
    return s


def write(svg, name):
    for chunk in svg.split("<text")[1:]:
        body = chunk.split(">", 1)[1].split("</text>")[0]
        assert "\n" not in body, f"{name}: newline inside a text element"
    (OUT / name).write_text(svg, encoding="utf-8", newline="\n")
    print(f"  wrote {name}")


def architecture(d):
    r, g, c = d["retrieval"], d["generation"], d["corpus"]
    W, H = 1060, 560
    s = header(W, H, "InsureAssist — system architecture",
               "Hybrid retrieval over three NFIP policy forms. Every label below is "
               "read from eval/reference_run.json.")

    s += '  <text class="lane" x="30" y="96">Ingestion — offline, run once</text>\n'
    s += box(30, 108, 190, 76, GREY_BG, HAIR, "Policy forms",
             [f"{c['documents']} NFIP SFIP documents"], f"{c['words']:,} words")
    s += arrow(224, 146, 258, 146)
    s += box(262, 108, 190, 76, BLUE_BG, BLUE, "Chunk",
             ["fixed window, selected on dev"],
             f"{r['chunking']['size']}/{r['chunking']['overlap']} chars")
    s += arrow(456, 146, 490, 146)
    s += box(494, 108, 200, 76, BLUE_BG, BLUE, "Embed",
             ["sentence-transformers"], r["dense_model"].split("/")[-1])
    s += arrow(698, 146, 732, 146)
    s += box(736, 108, 190, 76, GREEN_BG, GREEN, "Qdrant",
             ["vector database"], f"{c['chunks']} chunks")

    s += '  <text class="lane" x="30" y="246">Query — online, per question</text>\n'
    s += box(30, 258, 150, 74, GREY_BG, HAIR, "User", ["a policy question"])
    s += arrow(184, 295, 216, 295)
    s += box(220, 258, 160, 74, BLUE_BG, BLUE, "FastAPI", ["/ask endpoint"])

    # The two retrievers and the fusion that joins them: the design decision the
    # old diagram left out entirely.
    s += arrow(384, 278, 416, 246)
    s += arrow(384, 312, 416, 344)
    s += box(420, 210, 190, 68, GREEN_BG, GREEN, "Dense retrieval",
             ["cosine over Qdrant"], f"top {r['fusion']['candidate_depth_per_retriever']}")
    s += box(420, 312, 190, 68, AMBER_BG, AMBER, "BM25",
             [f"k1={r['bm25']['k1']}, b={r['bm25']['b']}"],
             f"top {r['fusion']['candidate_depth_per_retriever']}")
    s += arrow(614, 244, 648, 278)
    s += arrow(614, 346, 648, 312)
    s += box(652, 258, 170, 74, BLUE_BG, BLUE, "RRF fusion",
             [f"k = {r['fusion']['k']}, ranks only"], f"top {r['serving_top_k']} served")
    s += arrow(826, 295, 858, 295)
    s += box(862, 258, 168, 74, GREY_BG, HAIR, "Generator",
             ["answer with citations"], g["generator"])
    # Qdrant is queried by the dense retriever, not by the generator. The previous
    # vertical connector ran store -> model and implied the store fed it directly.
    s += arrow(766, 188, 620, 230)
    s += ('  <text class="cap" x="706" y="196" text-anchor="middle">queried by</text>\n')

    s += (f'  <line x1="30" y1="410" x2="{W-30}" y2="410" stroke="{HAIR}" '
          f'stroke-width="1"/>\n')
    s += '  <text class="lbl" x="30" y="436">What this diagram does not claim</text>\n'
    for i, ln in enumerate([
        f"The generator is {g['generator']} served through {g['backend']} and is " +
        "not fine-tuned. A LoRA notebook exists under archive/ and ships with " +
        "nothing.",
        "No LLM-as-judge metrics are published: only one local model is available, " +
        "and using it to grade its own output would be circular.",
        "There is no abstention rule. Retrieval always returns chunks, so " +
        "unanswerable questions reach the generator — see docs/LIMITATIONS.md.",
    ]):
        s += f'  <text class="cap" x="30" y="{458 + i*19}">{ln}</text>\n'
    s += "</svg>\n"
    write(s, "architecture.svg")


def lifecycle(d):
    r = d["retrieval"]
    W, H = 1060, 300
    s = header(W, H, "How a change reaches production",
               "The stages CI actually runs, in order. Fine-tuning is not one of "
               "them; it is archived.")
    stages = [
        ("1 · Ingest", ["corpus hashed and", "chunked into Qdrant"],
         "idempotent", BLUE_BG, BLUE),
        ("2 · Evaluate", ["retrieval scored on the", "frozen dev/test split"],
         f"{r['architecture']}", GREEN_BG, GREEN),
        ("3 · Verify", ["committed metrics must", "reproduce exactly"],
         "compare_runs", GREEN_BG, GREEN),
        ("4 · Containerize", ["image built and a live", "HTTP request served"],
         "Docker", BLUE_BG, BLUE),
        ("5 · Deploy", ["manifests validated", "against the cluster"],
         "Kubernetes", BLUE_BG, BLUE),
    ]
    x, w = 30, 190
    for i, (title, lines, mono, fill, edge) in enumerate(stages):
        s += box(x, 108, w, 92, fill, edge, title, lines, mono)
        if i < len(stages) - 1:
            s += arrow(x + w + 4, 154, x + w + 22, 154)
        x += w + 26

    s += (f'  <line x1="30" y1="228" x2="{W-30}" y2="228" stroke="{HAIR}" '
          f'stroke-width="1"/>\n')
    for i, ln in enumerate([
        "Every stage runs on each push through GitHub Actions. Step 3 is the one " +
        "that matters: retrieval is deterministic, so a drift means the corpus, " +
        "the labels, the chunking or the",
        "retriever changed without the committed artefact being regenerated. " +
        "Fine-tuning with LoRA and MLflow lives in archive/ and is not part of " +
        "this pipeline.",
    ]):
        s += f'  <text class="cap" x="30" y="{252 + i*19}">{ln}</text>\n'
    s += "</svg>\n"
    write(s, "mlops-pipeline.svg")


def main() -> int:
    d = json.loads(RUN.read_text(encoding="utf-8"))
    architecture(d)
    lifecycle(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
