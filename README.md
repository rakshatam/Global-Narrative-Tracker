# Global Narrative Tracker 🌍🤖

The **Global Narrative Tracker** is a high-performance, GPU-accelerated OSINT (Open Source Intelligence) pipeline. It autonomously tracks, clusters, and summarizes global news narratives in real-time. 

Powered by a native Docker Compose architecture, it leverages **PyTorch**, **HuggingFace Transformers**, and a **Cython-optimized Locality Sensitive Hashing (LSH)** engine to process massive amounts of unstructured web data at blinding speeds. 

It is also fully equipped with a production-grade **MLOps pipeline** featuring MLflow experiment tracking, Prometheus/Grafana GPU observability, and LanceDB serverless vector storage.

---

## 🏗️ System Architecture

The infrastructure relies on a strict, containerized microservice architecture built for maximum security and hardware utilization (NVIDIA GPU Passthrough).

```text
+-------------------------------------------------------------+
|                        Public Internet                      |
+------------------------------+------------------------------+
                               |
                   +-----------v-----------+
                   |   Cloudflare Tunnel   | (Optional)
                   +-----------+-----------+
                               |
                   +-----------v-----------+
                   |  NGINX Reverse Proxy  | 
                   |  (Cryptographic Auth) |
                   +----+-------------+----+
                        |             | (Subpaths /grafana, /mlflow)
           +------------v-+         +-v-------------+
           |   Next.js    |         |    FastAPI    |
           |  (Frontend)  |         |   (Backend)   |
           +--------------+         +-------+-------+
                                            |
                         +------------------v------------------+
                         |       PyTorch / HuggingFace         |
                         | (NVIDIA GPU Hardware Passthrough)   |
                         +--------+---------+--------+---------+
                                  |         |        |
             +--------------------+   +-----v----+   +--------------------+
             |                        |          |                        |
      +------v-------+         +------v------+   |                 +------v-------+
      |  Prometheus  | <------ |  LanceDB    |   +---------------> |    MLflow    |
      |  (Metrics)   |         | (Vector DB) |                     | (Exp. Logs)  |
      +------+-------+         +-------------+                     +--------------+
             |
      +------v-------+
      |   Grafana    |
      | (Dashboards) |
      +--------------+
```

---

## ⚡ NLP & Machine Learning Workflow

When a user initiates a topic search, the system executes a complex chronological pipeline:

```text
[ User Query ] 
      |
      v
+-----------------------+    Bypasses anti-bot measures to extract
|     Cloudscraper      | -> raw HTML from Bing News & targeted domains.
+---------+-------------+
          |
          v
+-----------------------+    Converts text into high-dimensional vector
|   all-MiniLM-L6-v2    | -> representations for semantic comparison.
+---------+-------------+
          |
          +--------------------------------------+
          |                                      |
          v                                      v
+-----------------------+              +-----------------------+
|  Cython MinHash LSH   |              |       LanceDB         |
| (In-Memory Clustering)|              | (Persistent Vectors)  |
+---------+-------------+              +-----------------------+
          |
          v
+-----------------------+    Generates a concise, highly accurate paragraph
|    distilbart-cnn     | -> summarizing the entire cluster of articles.
+---------+-------------+
          |
          v
+-----------------------+    Determines if the cluster's narrative is
|    bart-large-mnli    | -> Positive, Negative, or Neutral toward the topic.
+---------+-------------+
          |
          v
[ React Dashboard UI ]
```

---

## 📁 Codebase Breakdown

### Backend (`/backend`)
The nervous system of the application.
- **`main.py`**: The FastAPI server. Integrates Prometheus Instrumentator and LanceDB.
- **`ml_engine.py`**: Executes transformer model inference on the RTX GPU.
- **`cython_modules/lsh.pyx`**: A low-level C-extension that bypasses Python's GIL for blazing-fast clustering.
- **`db.py`**: SQLite database configuration. *(Note: We use a dual-database architecture. SQLite is strictly used for fast, relational CRUD operations like feeding the dashboard UI and storing timestamps/summaries. LanceDB handles the massive high-dimensional vector embeddings).*

### MLOps Infrastructure
- **Grafana & Prometheus**: Provide real-time observability. A custom backend gauge natively hooks into PyTorch to broadcast exact `torch.cuda` VRAM allocations to Grafana.
- **MLflow**: Tracks execution time, cluster counts, and hyperparameters for every single search query.
- **LanceDB**: A serverless, Apache Arrow-backed vector database. It continuously builds a permanent semantic memory of every article scraped. This enables the **Semantic History Search** feature, allowing users to query their offline memory via Cosine Similarity, bypassing web scraping entirely to instantly retrieve and cluster historical narratives.

### Frontend (`/frontend`)
- **`src/app/page.tsx`**: The main React interface built with Next.js and Tailwind CSS.
- **`next.config.ts`**: Configures the standalone Node server to proxy API requests internally.

### Infrastructure (Root)
- **`docker-compose.yml`**: The orchestration file linking the 6 microservices (NGINX, Frontend, Backend, Prometheus, Grafana, MLflow).
- **`nginx.conf`**: The security gatekeeper locking all dashboards (including MLOps tools) behind Basic Authentication.

---

## 🖥️ NVIDIA GPU Setup

Because this pipeline processes massive HuggingFace models, it requires a dedicated NVIDIA GPU.

### Windows (Recommended)
If you are on Windows 10/11, Docker Desktop WSL2 automatically handles GPU passthrough! There are no complicated toolkits to install.
1. Install the standard NVIDIA Game Ready or Studio Drivers on your host PC.
2. Install Docker Desktop.
3. The `docker-compose.yml` is already pre-configured to detect your RTX card natively.

### Linux (Ubuntu/Debian)
If you are deploying this on a headless Linux server, you must install the NVIDIA Container Toolkit first:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 🔒 Security Configuration

To prevent unauthorized API polling or dashboard access, the entire ecosystem (including Grafana and MLflow) is wrapped behind an **NGINX Reverse Proxy** with a strict Bcrypt Authentication wall enforced at the network layer.

---

## 🚀 Deployment & Installation

Disaster recovery and initial setup take only a single command:

```bash
docker-compose up -d --build
```
