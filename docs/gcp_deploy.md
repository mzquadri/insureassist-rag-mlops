# Phase 5 — Deploy to Google Cloud (GKE)

You'll put the app on **GKE** (managed Kubernetes), store artifacts in **Cloud Storage
(GCS)**, and host the container image in **Artifact Registry**.

> Cost: uses the **free $300 credit**. Delete the cluster at the end (last section) so you
> don't get charged after the trial.

## 0. One-time setup (you do this)
1. Create a Google Cloud account: https://cloud.google.com/free
2. Install the CLI: `winget install Google.CloudSDK` then restart the terminal.
3. Login and pick a project:
   ```powershell
   gcloud auth login
   gcloud projects create insureassist-<yourname> --name="InsureAssist"
   gcloud config set project insureassist-<yourname>
   gcloud auth configure-docker europe-west3-docker.pkg.dev
   ```
4. Enable the services:
   ```powershell
   gcloud services enable container.googleapis.com artifactregistry.googleapis.com storage.googleapis.com
   ```

## 1. Store the LoRA adapter + docs in Cloud Storage (GCS)
```powershell
gcloud storage buckets create gs://insureassist-<yourname> --location=europe-west3
gcloud storage cp -r finetune/adapter gs://insureassist-<yourname>/adapter
gcloud storage cp -r data gs://insureassist-<yourname>/data
```

## 2. Build + push the image to Artifact Registry
```powershell
gcloud artifacts repositories create insureassist --repository-format=docker --location=europe-west3

$IMG = "europe-west3-docker.pkg.dev/$(gcloud config get-value project)/insureassist/api:latest"
docker build -t $IMG .
docker push $IMG
```

## 3. Create the GKE cluster
```powershell
gcloud container clusters create-auto insureassist-cluster --location=europe-west3
gcloud container clusters get-credentials insureassist-cluster --location=europe-west3
```
(`create-auto` = GKE Autopilot: Google manages the nodes; you only pay for what pods use.)

## 4. Deploy
```powershell
# point the deployment image to your pushed image first (edit k8s/api-deployment.yaml
# 'image:' line to $IMG), then:
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml          # your filled-in copy of secret.example.yaml
kubectl apply -f k8s/qdrant.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/hpa.yaml
```

## 5. Get the public URL + test
```powershell
kubectl get service insureassist-api        # wait for EXTERNAL-IP
# then: curl http://EXTERNAL-IP/health
```

## 6. IMPORTANT — clean up to stop billing
```powershell
gcloud container clusters delete insureassist-cluster --location=europe-west3
gcloud storage rm -r gs://insureassist-<yourname>
```

## What you can now claim (truthfully)
Deployed a containerized RAG service to **GKE** with Cloud Storage artifacts, Artifact
Registry images, autoscaling (HPA), and a public inference endpoint.
