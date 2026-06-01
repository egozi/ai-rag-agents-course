# 🚀 Deploying Qdrant on Google Cloud Platform — Simple Demo Guide

---

## Prerequisites

- A Google account with GCP project created
- `gcloud` CLI installed locally → https://cloud.google.com/sdk/docs/install
- Basic terminal familiarity

---

## PART 1 — Local Machine: GCP Setup

### Step 1 — Authenticate and Set Project

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Step 2 — Enable Required API

```bash
gcloud services enable compute.googleapis.com
```

### Step 3 — Create the VM

```bash
gcloud compute instances create qdrant-server \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-ssd \
  --tags=qdrant-server \
  --metadata=enable-oslogin=true
```

### Step 4 — Open Firewall for Qdrant

```bash
gcloud compute firewall-rules create allow-qdrant \
  --allow=tcp:6333 \
  --target-tags=qdrant-server \
  --description="Qdrant REST API" \
  --source-ranges=0.0.0.0/0
```

### Step 5 — Get the VM's External IP

```bash
gcloud compute instances describe qdrant-server \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Save this IP — you'll need it at the end.

---

## PART 2 — Inside the VM

### Step 6 — SSH into the VM

```bash
gcloud compute ssh qdrant-server --zone=us-central1-a
```

### Step 7 — Install Docker

```bash
sudo apt-get update && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker --version
```

### Step 8 — Create Storage Directory

```bash
mkdir -p ~/qdrant/storage
```

### Step 9 — Generate and Save API Key

```bash
export QDRANT_API_KEY=$(openssl rand -hex 32)
echo "Your API key: $QDRANT_API_KEY"
echo "export QDRANT_API_KEY=$QDRANT_API_KEY" >> ~/.bashrc
```

> 📋 Copy this key — you'll need it in your `.env` file on your local machine.

### Step 10 — Run Qdrant

```bash
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v ~/qdrant/storage:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY \
  qdrant/qdrant:latest
```

### Step 11 — Verify Qdrant is Running

```bash
# Check container is up
docker ps

# Test health
curl http://localhost:6333/healthz

# Test with API key
curl http://localhost:6333/collections \
  -H "api-key: $QDRANT_API_KEY"
```

Expected response:
```json
{"result":{"collections":[]},"status":"ok","time":0.000123}
```

---

## PART 3 — Local Machine: Connect from Python

### Step 12 — Update your `.env` file

```bash
QDRANT_URL=http://YOUR_VM_EXTERNAL_IP:6333
QDRANT_API_KEY=your_generated_api_key_here
```

### Step 13 — Test the Connection

```python
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collections = client.get_collections()
print("✅ Connected!")
print(f"Collections: {collections}")
```

---

## Useful Commands

```bash
# View logs
docker logs qdrant -f

# Restart Qdrant
docker restart qdrant

# Check memory usage
docker stats qdrant --no-stream

# Access dashboard from local browser (SSH tunnel)
gcloud compute ssh qdrant-server --zone=us-central1-a \
  -- -L 6333:localhost:6333
# Then open: http://localhost:6333/dashboard
```

---

## Cost & Cleanup

```bash
# Stop VM between sessions (~$0/compute, ~$5/month disk only)
gcloud compute instances stop qdrant-server --zone=us-central1-a

# Start again
gcloud compute instances start qdrant-server --zone=us-central1-a

# Delete everything (zero charges)
gcloud compute instances delete qdrant-server --zone=us-central1-a
gcloud compute firewall-rules delete allow-qdrant
```

> 💡 New GCP accounts get **$300 free credits** — more than enough for this demo.
