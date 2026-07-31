# Global Narrative Tracker: Complete System Documentation

This document serves as the master operations manual for the Global Narrative Tracker. It covers the system architecture, day-to-day operations, disaster recovery, security configuration, and strategies for public deployment.

## 1. System Architecture & MLOps

The system operates on a highly optimized, GPU-accelerated Docker Compose stack consisting of 6 services:

### A. Core Services
- **Reverse Proxy (NGINX)**: Acts as the gatekeeper and router. Enforces Bcrypt Basic Authentication on Port 80.
- **Frontend (Next.js)**: Runs in lightweight `standalone` mode (1GB RAM cap) to serve the React UI.
- **Backend (FastAPI + PyTorch)**: The ML Engine linked to the host's NVIDIA RTX GPU. Runs web scraping, Cython LSH clustering, and HuggingFace models. *(Note: We use a dual-database architecture. SQLite handles fast relational UI data, while LanceDB handles the massive vector embeddings).*

### B. MLOps Integrations
- **LanceDB (Serverless Vector Storage)**: Runs internally within the FastAPI backend. Flushes all 384-dimensional PyTorch embeddings and metadata into a persistent Apache Arrow database. This powers the **Semantic History Search** dashboard button, which lets you instantly query and cluster your historical offline articles using mathematical Cosine Similarity thresholds.
- **Prometheus & Grafana (Observability)**: Prometheus scrapes the FastAPI backend every 5 seconds. Grafana visualizes PyTorch GPU VRAM consumption and API latency in real-time. Accessible securely at `http://localhost/grafana`.
- **MLflow (Experiment Tracking)**: Tracks the hyperparameters, scraped article counts, and execution speeds of every search query. Accessible securely at `http://localhost/mlflow`.

---

## 2. Credentials & Security

The system is secured by a static NGINX `.htpasswd` file using military-grade Bcrypt hashes. Because NGINX proxies all traffic, Grafana and MLflow are completely protected by this wall.

> **Main Dashboard Credentials (NGINX):**
> - **Username:** `qwerty`
> - **Password:** `200308`

> **Grafana Internal Credentials:**
> - **Username:** `admin`
> - **Password:** `200308`

### How to Change the Main Password
You must generate a new mathematical Bcrypt hash:
1. Open your terminal in your project folder (`C:\Users\Lenovo\Downloads\project`).
2. Run this exact command to generate a Bcrypt hash:
   ```cmd
   docker run --rm httpd:alpine htpasswd -B -n -b YourNewUsername YourNewPassword > htpasswd
   ```
3. Restart NGINX to apply:
   ```powershell
   docker-compose restart nginx
   ```

---

## 3. Container Lifecycle Management

Always run these commands from inside `C:\Users\Lenovo\Downloads\project`.

### Pausing & Resuming
If you are playing a heavy video game or rendering a video and need to free up your RTX GPU and RAM:
- **To Pause:** `docker-compose stop`
- **To Resume:** `docker-compose start`

### Complete Shutdown
- **To Shut Down:** `docker-compose down`
- **To Boot Up:** `docker-compose up -d`

### Dashboard Purge Controls
- **Purge Database & Clusters:** Wipes the live SQLite feed and in-memory Cython clusters, but *keeps* the LanceDB offline vectors safe.
- **Flush LanceDB Vectors:** Permanently deletes your offline semantic memory dataset. Use with caution!

---

## 4. Disaster Recovery

If you accidentally delete all your Docker images or move to a new laptop, you can recover the entire pipeline with one command:

```powershell
docker-compose up -d --build
```

**What this command does:**
1. Reinstalls PyTorch, LanceDB, Cython, and all Python requirements in the backend.
2. Compiles the Next.js production frontend.
3. Boots Prometheus, Grafana, and MLflow, pulling their latest official images.
4. Mounts your persistent volumes (`project_ml_cache`, `mlflow_data`, `grafana_provisioning`) so no historical data or machine learning models are lost.

---

## 5. NVIDIA GPU Configuration

To ensure the backend runs at blazing speeds, the Docker container hooks directly into your host NVIDIA GPU.

**On Windows:** 
Docker Desktop with WSL2 handles GPU passthrough automatically. Just ensure your standard NVIDIA Windows drivers are up-to-date and Docker Desktop is running. No extra toolkits are required.

**On Linux:**
You must install the `nvidia-container-toolkit` before running `docker-compose up`:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 6. Managing Disk Space (Docker VHDX Bug)

When you frequently rebuild Docker images, the Docker WSL2 backend hoards disk space inside a virtual hard disk (`.vhdx`) and refuses to release it back to the Windows `C:` drive even after you delete the images.

Here is the master cheat sheet to safely reclaim your space without losing your LanceDB offline vectors or MLflow tracking data:

1. **Prune the Cache (Safe)**
   Run this in your normal terminal. *(⚠️ WARNING: Do NOT use the `--volumes` flag!)*
   ```powershell
   docker system prune -a
   ```

2. **Physically Compact the VHDX**
   Quit Docker Desktop. Open an **Administrator Command Prompt** and run:
   ```cmd
   wsl --shutdown
   diskpart
   ```
   Then paste these commands inside the `DISKPART>` prompt:
   ```text
   select vdisk file="C:\Users\Lenovo\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
   attach vdisk readonly
   compact vdisk
   detach vdisk
   exit
   ```
   Your `C:` drive space will instantly return!

---

## 7. How to Make it Online (Publicly Accessible)

You can use Cloudflare Tunnels to safely route internet traffic to your local RTX machine without opening router ports.

1. Download **Cloudflared** into your project folder.
2. Run this exact command in your terminal to start the tunnel: 
   ```powershell
   cd C:\Users\Lenovo\Downloads\project
   cloudflared tunnel --url http://localhost:80
   ```
3. Share the temporary `https://...trycloudflare.com` URL with anyone!
