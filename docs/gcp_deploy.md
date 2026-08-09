# Deploying to Google Cloud (GKE)

This deploys the whole system — Qdrant, Ollama (the LLM), and the API — to a managed
**GKE Autopilot** cluster, with a public endpoint. It's self-contained: no external API
keys needed.

> Uses the **$300 free credit**. Delete the cluster at the end (last section) so you don't
> get charged after the trial. A short setup + test session costs only a few cents.

Set a few shell variables first (PowerShell):
```powershell
$PROJECT = "insureassist-<yourname>"     # must be globally unique
$REGION  = "europe-west3"
$REPO    = "insureassist"
$IMG     = "$REGION-docker.pkg.dev/$PROJECT/$REPO/api:latest"
```

## 1. Account & project (one-time)
```powershell
gcloud auth login
gcloud projects create $PROJECT
gcloud config set project $PROJECT
# link billing (needed even for the free tier): easiest via the Cloud Console UI,
# Billing -> link the project to your billing account.
gcloud services enable container.googleapis.com artifactregistry.googleapis.com
```

## 2. Build & push the image
```powershell
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
docker build -t $IMG .
docker push $IMG
```

## 3. Create the cluster
```powershell
gcloud container clusters create-auto insureassist --location=$REGION
gcloud container clusters get-credentials insureassist --location=$REGION
```

## 4. Deploy
```powershell
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/qdrant.yaml
kubectl apply -f k8s/ollama.yaml

# point the API + ingest Job at the pushed image, then deploy them:
kubectl set image -f k8s/api-deployment.yaml api=$IMG --local -o yaml | kubectl apply -f -
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/hpa.yaml

# wait for Ollama, pull the model, then load the documents:
kubectl rollout status deploy/ollama
kubectl exec deploy/ollama -- ollama pull llama3.2:3b

kubectl set image -f k8s/ingest-job.yaml ingest=$IMG --local -o yaml | kubectl apply -f -
kubectl wait --for=condition=complete job/ingest --timeout=300s
```

## 5. Get the public URL & test
```powershell
kubectl get service insureassist-api      # wait for EXTERNAL-IP
curl http://EXTERNAL-IP/health
curl -X POST http://EXTERNAL-IP/ask -H "Content-Type: application/json" -d "{\"question\":\"Does home insurance cover a burst pipe?\"}"
```

## 6. Clean up (stops billing)
```powershell
gcloud container clusters delete insureassist --location=$REGION
gcloud artifacts repositories delete $REPO --location=$REGION --quiet
```
