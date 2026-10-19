# Kubernetes run evidence

Raw output from applying `k8s/` to a local kind cluster, kept as captured.
[`../KUBERNETES_RUN.md`](../KUBERNETES_RUN.md) is the write-up, including the
three defects the run found.

| File | What it is |
|---|---|
| `cluster-state.txt` | `kubectl get pods -o wide`, `get svc`, `get statefulset,deployment,job` |
| `http-checks.txt` | `GET /health`, `GET /ready`, `POST /ask` against the API Service |
| `retrieval-in-pod.txt` | hybrid retrieval run inside the API pod against the in-cluster Qdrant |

Two things to read carefully rather than skim.

**`qdrant-0` shows 7 restarts.** Those are the CrashLoopBackOff described as
defect 1, before the snapshots path was fixed. The pod has been stable since.

**`POST /ask` returns 503, and that is the expected result here.** Ollama is not
deployed - this machine has 7.7 GB of RAM - so the generator is genuinely absent
and the service says which dependency is missing instead of answering anyway.
Retrieval is therefore verified in `retrieval-in-pod.txt` instead, which exercises
the embedder, the BM25 index, Qdrant over cluster DNS and the RRF fusion.

**The top retrieval hit is from the wrong form.** The question names the Dwelling
Form; the first result is the condominium form. Right provision, wrong document -
the failure mode `docs/BENCHMARK.md` already measures as top-document accuracy
0.556. It is left in because it is what the system does.
