# The ask flow as a LangGraph state machine

`src/graph.py` expresses `retrieve -> decide -> answer or abstain` as an explicit
graph. `src/rag.py:answer()` already does this as a straight-line function, the
API still calls that one, and this module does not replace it.

What the graph adds is that **the abstention policy becomes a swappable node**.
That matters here because which policy to ship was a real question in this
repository, it was answered with evidence, and the answer was counter-intuitive.

## The decision it encodes

`eval/abstention_threshold.json` tested whether a similarity threshold is a
defensible abstention rule: choose it on dev by Youden's J, apply it unchanged to
test.

| | dev | test |
|---|---|---|
| balanced accuracy | 0.8016 | **0.5556** |
| Youden's J | 0.6032 | **0.1111** |

Always answering scores 0.5 balanced accuracy by construction, so on held-out
data the threshold buys **0.0556** having looked worth 0.30 on dev. The reason is
in the same artefact: **17 of 18 unanswerable questions score above the weakest
answerable one**, so the signal barely separates the classes.

So `POLICY_NO_EVIDENCE` - abstain only when retrieval returned nothing - is what
ships, and `POLICY_THRESHOLD` is kept runnable rather than only described.

## It reproduces the study exactly

`eval/graph_abstention.py` runs both policies through the graph over all 50
questions. Every field matches the earlier direct measurement:

| threshold, test split | `abstention_threshold.json` | via the graph |
|---|---|---|
| sensitivity | 0.5556 | 0.5556 |
| specificity | 0.5556 | 0.5556 |
| balanced accuracy | 0.5556 | 0.5556 |
| Youden's J | 0.1111 | 0.1111 |
| false abstentions | 8 | 8 |
| false answers | 4 | 4 |

And `POLICY_NO_EVIDENCE` lands on sensitivity 1.0, specificity 0.0, balanced
accuracy 0.5 - identical to that file's `test_baseline_always_answer` row, which
is the arithmetic consequence of never abstaining when something was retrieved.

That agreement is the point of the evaluation. If routing the same decision
through four nodes and a conditional edge had changed the numbers, the graph
would not be a faithful restatement and the difference would be a defect.

**One convention had to be fixed to make this comparison honest.** The first
version of the evaluation treated *unanswerable* as the positive class. Balanced
accuracy and Youden's J are symmetric under swapping the classes, so both
matched immediately - while sensitivity and specificity were silently
transposed, 0.8889 against 0.7143 on dev. It agreed on the headline numbers and
disagreed on every row. The positive class is now *answerable*, matching the
earlier file.

## Running it

```bash
pip install -r requirements-graph.txt
python -m eval.graph_abstention --out eval/graph_abstention.json
```

Needs a populated Qdrant, because the decision reads retrieval's output. The run
recorded in `eval/graph_abstention.json` used the Qdrant deployed on the kind
cluster described in [`KUBERNETES_RUN.md`](KUBERNETES_RUN.md), reached over a
port-forward - reads are small, unlike the bulk ingest that had to run in-cluster.

**No LLM is involved, and that is asserted rather than assumed.** The graph is
compiled with `decide_only=True`, which ends it after the decision node, and the
generator handed in raises `AssertionError` if called. The first attempt did not
have `decide_only` and the assertion fired immediately - correctly, because an
"answer" decision routes into `generate`, so evaluating the *decision* on the
full graph would have paid for output it discards.

## Why langgraph is an optional dependency

It is in `requirements-graph.txt`, not `requirements.txt`:

- `src/api.py` does not import `src.graph`, and `tests/test_graph.py` asserts
  that as a source property rather than by import, which would pass either way
  in an environment that happens to have langgraph installed.
- `build_graph()` raises `ImportError` naming the requirements file, at call time
  rather than import time, so the service starts normally without it.
- Installing it **downgrades `websockets` from 17.2 to 16.1.1**, because
  langgraph-sdk caps it. `uvicorn[standard]` also depends on websockets, so the
  downgrade reaches the service's own tree. That is a second reason to keep it
  out of the default install.

`tests/test_graph.py` covers the routing, both policies, the tie-break at exactly
the threshold, missing dense scores, and the unknown-policy guard - 11 tests, no
network and no LLM, skipped cleanly when langgraph is absent.
