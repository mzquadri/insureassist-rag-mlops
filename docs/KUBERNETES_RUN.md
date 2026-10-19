# Applying the manifests, and the three defects that found

The manifests in `k8s/` had never been applied. `.github/workflows/deploy.yml` is
`workflow_dispatch`-only, targets GKE, and says so itself: "Until then, this file
simply documents the pipeline (it won't run without the secrets)." So the YAML
was reviewed but never executed.

Running it on a local [kind](https://kind.sigs.k8s.io/) cluster found three
defects. One of them meant **Qdrant could never have started under these
manifests at all**, on any cluster.

## Defect 1: Qdrant crashlooped, and the HTTP API never bound

`k8s/qdrant.yaml` sets `runAsNonRoot: true` with `runAsUser: 1000`. Qdrant's
default storage and snapshot paths are `./storage` and `./snapshots`, relative to
its working directory `/qdrant` - which belongs to the image and is owned by
root. The PersistentVolumeClaim is mounted over `/qdrant/storage`, so that one
happened to be writable. **Nothing was mounted at `/qdrant/snapshots`**, and uid
1000 cannot create a directory there.

The symptom was almost unreadable:

```
INFO  qdrant::tonic: Qdrant gRPC listening on 6334
ERROR qdrant::startup: Panic backtrace:
...
ERROR qdrant::startup: Panic occurred in file src/main.rs at line 823
```

gRPC came up on 6334; the HTTP API on 6333 never did. So the readiness probe got
`connection refused` indefinitely and the pod sat in `CrashLoopBackOff`:

```
Warning  Unhealthy  kubelet  Readiness probe failed: Get "http://10.244.0.6:6333/readyz":
                             dial tcp 10.244.0.6:6333: connect: connection refused
```

**It was not a memory problem**, which was the first guess on a 7.7 GB machine.
The container terminated with `exitCode: 101, reason: Error` - Rust's panic exit,
not the `137`/`OOMKilled` a memory kill produces - and the node reported
`MemoryPressure False` with 802Mi of 4Gi requested throughout.

The only other clue was a warning about an unrelated file, which is what made
this take three attempts to find:

```
WARN qdrant::startup: Failed to create init file indicator:
     failed to create file `.qdrant-initialized`: Permission denied (os error 13)
```

That one is genuinely only a warning. Fixing it changed nothing.

**The fix** points both paths inside the mounted volume:

```yaml
env:
  - name: QDRANT__STORAGE__STORAGE_PATH
    value: /qdrant/storage/collections
  - name: QDRANT__STORAGE__SNAPSHOTS_PATH
    value: /qdrant/storage/snapshots
```

```
INFO qdrant::actix: Qdrant HTTP listening on 6333
```

`/readyz` returns 200 and `/collections` answers. Verified by re-applying the
committed file rather than the patch used while diagnosing.

**A wrong fix, recorded because it fails in a confusing way.** Setting
`workingDir` to move the defaults instead breaks the container before it starts:
the image's entrypoint is `./entrypoint.sh`, so a different working directory
makes it unresolvable and the container never runs at all.

```
exitCode: 128, reason: StartError
unable to start container process: exec: "./entrypoint.sh":
stat ./entrypoint.sh: no such file or directory
```

## Defect 2: the image carried the whole CUDA stack, unused

`torch` is not in `requirements.txt`; it arrives transitively because
sentence-transformers needs it. On Linux, PyPI's default torch wheel is the CUDA
build, so `pip install -r requirements.txt` pulled the NVIDIA runtime with it:

```
Downloading nvidia_cublas-13.1.1.3-...whl (423.1 MB)
Downloading nvidia_cufft-12.0.0.61-...whl (214.1 MB)
Downloading nvidia_cuda_nvrtc-13.0.88-...whl (90.2 MB)
... nvshmem, cuDNN, NCCL, cuSPARSE, cuSOLVER, cuRAND, cuFile, cuPTI
```

Several gigabytes of accelerator libraries, in an image whose own README says
"Nothing here needs a GPU or a network beyond pip", to serve a 33M-parameter
embedding model from the CPU.

**The obvious fix does not work.** Installing CPU torch first and then
`requirements.txt` leaves the second command free to re-resolve the whole graph,
decide sentence-transformers wants torch from the default index, and replace the
CPU build. The NVIDIA wheels reappear in the second step having been absent from
the first:

```
#11 Successfully installed ... torch-2.14.1+cpu     <- first step, clean
#10 Downloading nvidia_cufft-12.0.0.61 (214.1 MB)   <- second step, back again
```

`--extra-index-url` is not enough either: an extra index is a fallback consulted
*in addition to* the default, and with both offering torch, PyPI's wheel can
still win.

**The fix** is one resolve with the CPU index as the primary and PyPI as the
fallback:

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple
```

Zero NVIDIA wheels, and the dependency install dropped from 2,779 seconds and
still running to **1,053 seconds**.

## Defect 3, a documentation one

`k8s/README.md` describes `qdrant.yaml` as "the vector DB (Deployment +
Service)". It is a StatefulSet with a `volumeClaimTemplate`, and the manifest's
own header explains why that matters - as a Deployment "any restart destroyed the
entire index". The README was not updated when that was fixed.

## Getting kind to run at all

Not a defect in this repository, but necessary to reproduce the above.

kind 0.33.0's default node image is Kubernetes v1.37, and `kubeadm init` fails at
`wait-control-plane` on this machine. Docker Desktop's WSL2 backend runs
**cgroup v1**:

```
$ docker info | grep -i cgroup
 Cgroup Driver: cgroupfs
 Cgroup Version: 1
 Kernel Version: 5.15.167.4-microsoft-standard-WSL2
```

kind warns about it directly - "cgroup v1 is deprecated in Kubernetes and will
not be supported in a future kind release" - and v1.37 no longer bootstraps
there. Pinning an older node image works and needs no change to the host's WSL
configuration:

```sh
kind create cluster --name insureassist --image kindest/node:v1.31.0 --wait 300s
# Ready after 1m35s
```

## What was run

```sh
kind create cluster --name insureassist --image kindest/node:v1.31.0 --wait 300s
docker build -t insureassist:v0.2.0 .
kind load docker-image insureassist:v0.2.0 --name insureassist

kubectl apply -f k8s/configmap.yaml -f k8s/qdrant.yaml
kubectl rollout status statefulset/qdrant

kubectl apply -f k8s/ingest-job.yaml
kubectl wait --for=condition=complete job/ingest --timeout=30m

kubectl apply -f k8s/api-deployment.yaml -f k8s/api-service.yaml
kubectl scale deployment/insureassist-api --replicas=1
```

## Deliberate deviations, and why

**Ollama is not deployed.** `k8s/ollama.yaml` was left unapplied. The model it
pulls is `llama3.2:3b`, and this machine has 7.7 GB of RAM with well under 1 GB
free; several containers were OOM-killed (`Exited (137)`) during earlier work in
this session. Consequence, stated plainly: **`POST /ask` cannot generate an
answer in this deployment.** The retrieval half is verified instead, and how is
described below.

**One API replica instead of two.** `api-deployment.yaml` requests 1Gi per pod
and asks for two. The manifest is left at 2, which is a reasonable default for a
real cluster; the running deployment was scaled to 1 for the same memory reason.

**The HPA was not applied.** `hpa.yaml` scales on CPU, which needs
metrics-server, which kind does not install. Applying it would have produced an
HPA stuck at `<unknown>/70%` - worse than not applying it, because it looks like
a working autoscaler.

**Ingestion runs in-cluster, not over a port-forward.** Loading the corpus
through `kubectl port-forward` does not work here: the Qdrant client timeout is
hardcoded to 120 seconds (`src/providers.py:93`) against `BATCH_SIZE = 128`
(`src/ingest.py:37`), and exactly one batch landed - 128 of 314 points - before
timing out. A port-forward funnels bulk writes through a single stream via the
API server. `ingest-job.yaml` talks to the `qdrant` Service directly over cluster
DNS, which is what it is for, and completed the whole corpus.

## What the run produced

Captured in [`k8s-evidence/`](k8s-evidence/) as written.

### Everything scheduled and stayed up

```
NAME                                READY   STATUS      RESTARTS   AGE
ingest-xsxt7                        0/1     Completed   0          35m
insureassist-api-7b9698c89f-9rmx5   1/1     Running     0          116s
qdrant-0                            1/1     Running     7 (9h ago)  11h

job.batch/ingest   Complete   1/1   32m
```

Qdrant's 7 restarts are the CrashLoopBackOff from defect 1, before the fix. The
image is **453 MB**, and `torch 2.14.1+cpu` with `torch.version.cuda` of `None`
inside it, which is defect 2 fixed.

### Ingestion

```
Collection 'nfip_sfip' ready (dim=384).
  nfip-sfip-dwelling: 111 chunks
  nfip-sfip-general-property: 99 chunks
  nfip-sfip-rcbap: 104 chunks
Ingested 314 chunks from 3 documents.
Collection now holds 314 points, one per chunk.
```

314 is the count `docs/BENCHMARK.md` and `eval/retrieval_config.json` both record
for 800/120, so the cluster chunked the corpus the same way every other run has.

### The probes

```
GET /health  ->  200  {"status":"ok"}
GET /ready   ->  200  {"status":"ready","dependencies":{
                        "embedder":{"ready":true},
                        "lexical_index":{"ready":true},
                        "vector_store":{"ready":true}}}
```

`/ready` is the one that matters: it reports the API pod reaching the `qdrant`
Service over cluster DNS, not just that the process is alive.

### /ask without a generator, and retrieval with one

`POST /ask` returns **503** and names what is missing rather than improvising:

```json
{"status":"dependency_unavailable",
 "detail":"generator 'llama3.2:3b' at http://ollama:11434 did not answer",
 "dependency":"generator"}
```

That is the intended behaviour of a deployment without Ollama, and it is the
reason retrieval is verified separately. There is no retrieval-only endpoint, so
it was run inside the pod against the in-cluster Qdrant:

```
$ kubectl exec insureassist-api-... -- python -c "from src.rag import retrieve; ..."
question: How long do I have to send a proof of loss after a flood under the Dwelling Form?
contexts returned: 5
  nfip-sfip-rcbap#d026af40c528             rrf=0.032018  dense=0.8282
  nfip-sfip-dwelling#e5de6486dc67          rrf=0.031514  dense=0.8157
  nfip-sfip-general-property#823f35c2549f  rrf=0.031514  dense=0.7922
  nfip-sfip-dwelling#b1e73cc5a690          rrf=0.031319  dense=0.7887
  nfip-sfip-rcbap#032a6c69a1a0             rrf=0.030366  dense=0.7763

top chunk offsets: 45560 - 46360
"ase of Loss / In case of a flood loss to insured property, you must: /
 1. Give prompt written notice to us. ..."
```

Both halves of the hybrid retriever ran: the RRF scores are fused ranks and the
dense cosines come from the embedder in the pod.

**Read the top hit honestly.** The question says "under the Dwelling Form" and
the top result is from **rcbap**, the condominium form. The right *provision* -
Case of Loss, with the proof-of-loss requirement - in the wrong *document*. That
is exactly the failure mode `docs/BENCHMARK.md` measures as top-document accuracy
0.556 and attributes to the three forms being near-duplicates. The cluster run
reproduced the known weakness rather than hiding it.
