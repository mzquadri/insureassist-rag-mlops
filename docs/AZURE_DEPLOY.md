# Deploying to Azure Container Apps

**Only the API container is hosted on Azure. Nothing else is.** There is no
Qdrant and no generator in this resource group, deliberately, because each would
add cost for no additional thing demonstrated. The consequences, stated up front
rather than discovered by trying it:

- `GET /health` returns **200**. The service is up and serving.
- `GET /ready` returns **503**, and it is correct to. The vector store is not
  deployed here, so the readiness check reports `vector_store: ready: false`
  while confirming the embedder and the BM25 index are both ready.
- `POST /ask` cannot answer. It needs both the vector store and a generator.

**Full retrieval is not demonstrated on Azure, and is not meant to be.** It runs
locally and on a local kind cluster: see [`KUBERNETES_RUN.md`](KUBERNETES_RUN.md),
where the whole stack is deployed, the corpus is ingested to 314 points, and
hybrid retrieval is executed inside the API pod against an in-cluster Qdrant.

What this deployment shows is the containerised service running on managed
infrastructure, reachable over the public internet, scaling to zero when idle.

The API runs on Azure Container Apps, scaling to zero when idle.

**URL:** <https://insureassist-api.ashytree-3df1e747.francecentral.azurecontainerapps.io>

```
$ curl https://insureassist-api.ashytree-3df1e747.francecentral.azurecontainerapps.io/health
{"status":"ok"}
HTTP 200  total 2.218s     <- cold start, scaled up from zero replicas
HTTP 200  total 0.636s     <- warm
```

Raw output in [`azure-evidence/`](azure-evidence/). Deployed 19 October 2026.

## What is deployed, and what is not

| | |
|---|---|
| Container app | `insureassist-api`, 1 vCPU / 2.0 GiB, min replicas **0**, max 1 |
| Image | `cacd7b38a858acr.azurecr.io/insureassist:v0.2.0`, digest `sha256:39df9471a168…` |
| Environment | `insureassist-env`, workload profile type **Consumption** |
| Region | **France Central** |
| Resource group | `rg-insureassist` (one group, everything in it) |

**Qdrant and Ollama are not deployed.** Only the API is, so:

```
GET /health  ->  200  {"status":"ok"}
GET /ready   ->  503  {"status":"not_ready","dependencies":{
                        "embedder":{"ready":true},
                        "lexical_index":{"ready":true},
                        "vector_store":{"ready":false,"detail":"ResponseHandlingException"}}}
```

That 503 is the correct answer, not a failure of the deployment. `/health` is
liveness and says only that the process is up; `/ready` checks the dependencies
and reports which one is missing. The embedder and the BM25 index both come up
inside the container; the vector store does not exist in this resource group.

A full deployment would add Qdrant with a persistent volume and a generator. The
Kubernetes manifests in `k8s/` do that, and `docs/KUBERNETES_RUN.md` is the run
where retrieval was verified end to end.

## Two things about this subscription that shaped the result

Both are properties of an **Azure for Students** subscription, found by hitting
them, and both are worth knowing before planning a deployment on one.

**1. Server-side container builds are not available.**

`az containerapp up --source .` and `az acr build` both fail:

```
ERROR: (TasksOperationsNotAllowed) ACR Tasks requests for the registry
cacd7b38a858acr and aaecd3e1-... are not permitted.
```

ACR Tasks is what performs a build inside Azure, so neither the one-command
deploy nor a standalone remote build works. The image has to be **built locally
and pushed**, which is what the commands below do. This also explains an earlier
symptom that looked like a hang: `az acr task list-runs` reported no runs,
because the request was rejected before a run was ever created.

**2. Only five regions are permitted.**

An `Allowed resource deployment regions` policy is assigned to the subscription:

```
listOfAllowedLocations = [polandcentral, austriaeast, denmarkeast,
                          francecentral, swedencentral]
```

Neither `westeurope` nor `germanywestcentral` is in it, and attempting either
fails at the Log Analytics workspace with `RequestDisallowedByAzure`. The policy
message does not name the allowed set, so it has to be read off the assignment.
**France Central** was chosen as the closest permitted region to Germany; it is
also marginally cheaper than Sweden Central on the environment meter ($0.125
against $0.130 per hour).

## The commands that produced it

```sh
# One-time, free: the CLI extension and the resource providers
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.OperationalInsights

# A $10/month budget before anything billable, alerting at 50/90/100%
az rest --method put \
  --url "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Consumption/budgets/insureassist-10usd?api-version=2023-05-01" \
  --body @budget.json

az group create --name rg-insureassist --location francecentral

# Creates the environment, the Log Analytics workspace and the registry.
# The build step inside it FAILS on this subscription (ACR Tasks), but the
# three resources it creates first are the ones needed below.
az containerapp up --name insureassist-api --resource-group rg-insureassist \
  --location francecentral --environment insureassist-env \
  --source . --ingress external --target-port 8000

# Build locally and push, since ACR Tasks cannot build server-side
docker build -t insureassist:v0.2.0 .
az acr update -n cacd7b38a858acr -g rg-insureassist --admin-enabled true
docker login cacd7b38a858acr.azurecr.io -u cacd7b38a858acr -p "$ACR_PASSWORD"
docker tag insureassist:v0.2.0 cacd7b38a858acr.azurecr.io/insureassist:v0.2.0
docker push cacd7b38a858acr.azurecr.io/insureassist:v0.2.0

az containerapp create \
  --name insureassist-api --resource-group rg-insureassist \
  --environment insureassist-env \
  --image cacd7b38a858acr.azurecr.io/insureassist:v0.2.0 \
  --registry-server cacd7b38a858acr.azurecr.io \
  --registry-username cacd7b38a858acr --registry-password "$ACR_PASSWORD" \
  --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 1 \
  --cpu 1.0 --memory 2.0Gi \
  --env-vars QDRANT_URL=http://unset:6333 QDRANT_COLLECTION=nfip_sfip \
             EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 LLM_BACKEND=ollama \
             OLLAMA_URL=http://unset:11434 CORPUS=nfip

curl https://insureassist-api.ashytree-3df1e747.francecentral.azurecontainerapps.io/health
```

`az acr login` failed where `docker login` with the admin credential succeeded,
which is why the admin user is enabled above. **The registry password is not in
this file and must not be committed**; read it with
`az acr credential show -n cacd7b38a858acr -g rg-insureassist --query "passwords[0].value" -o tsv`.

## Cost

Per-unit prices from the Azure retail API for `francecentral`, at deployment:

| Meter | Price |
|---|---|
| Container Apps vCPU, active | $0.000024 / vCPU-second |
| Container Apps memory, active | $0.000003 / GiB-second |
| Container Apps requests | $0.40 / 1M |
| Environment Management Hour | $0.125 / hour |
| ACR Basic registry unit | $0.1666 / day |
| ACR Basic data stored | $0.10 / GB / month |

`min-replicas 0` is what keeps this small: with no traffic there is no replica
and no vCPU or memory charge, and a cold start costs about two seconds on the
first request. The fixed costs are the registry at $0.1666/day and, if the
`Environment Management Hour` meter applies to a Consumption-only environment,
$0.125/hour.

Whether that meter applies is the one thing the price list cannot settle. The
environment here reports `workloadProfileType: Consumption`, and that meter is
documented against *Dedicated* workload profiles, so it should not be charged -
but the actual charges are checked rather than assumed, and
[`azure-evidence/charges.txt`](azure-evidence/charges.txt) records what Azure
actually billed.

## Tearing it down

Everything is in one resource group, so one command removes all of it:

```sh
az group delete --name rg-insureassist --yes
```

The `insureassist-10usd` budget is a subscription-level resource and survives
that, deliberately.
