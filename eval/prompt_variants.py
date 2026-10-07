"""Prompt arms for the citation experiment.

`eval/citation_structure.json` records that 5 of 18 answers cited a context block that
was never supplied - `[6]`, `[7]` and `[16]` against five numbered blocks. The served
prompt asks the model to "cite the numbers you used in square brackets" and never says
which numbers exist, so the obvious question is whether that is a prompt defect or a
model limitation.

This holds the two arms. `served` is imported from `src.rag` rather than copied, so the
control is literally what the service sends and cannot drift from it. `ranged` changes
one clause and nothing else.

Nothing here is wired into the service. A prompt that scores better on eighteen
questions is a measurement, not a reason to change what is served, and generation here
is only near-deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.rag import build_prompt

Prompt = Callable[[str, Sequence[dict]], str]


def ranged_prompt(question: str, contexts: Sequence[dict]) -> str:
    """The served prompt, with the valid citation range stated.

    The only difference from `build_prompt` is the clause naming 1..n. Keeping the rest
    byte-identical is what makes the comparison attributable to that clause.
    """
    context_block = "\n\n".join(
        f"[{index}] Source: {c['source']} ({c['cfr_citation']})\n{c['text']}"
        for index, c in enumerate(contexts, start=1)
    )
    return (
        "You are an insurance policy assistant. Answer the question using ONLY the "
        f"context below. The context blocks are numbered 1 to {len(contexts)}; cite "
        "only those numbers, in square brackets. If the answer is not in the context, "
        "say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


#: The arms, by name. `served` is the real one; see the module docstring.
VARIANTS: dict[str, Prompt] = {
    "served": build_prompt,
    "ranged": ranged_prompt,
}


__all__ = ["VARIANTS", "Prompt", "ranged_prompt"]
