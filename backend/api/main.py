from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import hashlib
import uuid
import numpy as np
import time

# Adjust path to import ml_engine and cython_modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine import get_stance, get_embedding, summarize_text
from search_engine import search_and_scrape
import db

# MLOps Imports
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import mlflow
import lancedb
import pyarrow as pa
import torch

app = FastAPI(title="Global Narrative Tracker API")

# Start Prometheus Instrumentator
Instrumentator().instrument(app).expose(app)

# --- MLOps: Prometheus Metrics ---
gpu_memory_allocated = Gauge('gpu_memory_allocated_bytes', 'GPU memory currently allocated by PyTorch')
gpu_memory_reserved = Gauge('gpu_memory_reserved_bytes', 'GPU memory currently reserved by PyTorch caching allocator')

def update_gpu_metrics():
    if torch.cuda.is_available():
        gpu_memory_allocated.set(torch.cuda.memory_allocated(0))
        gpu_memory_reserved.set(torch.cuda.memory_reserved(0))

@app.on_event("startup")
async def startup():
    
    # Initialize MLflow (ensure MLflow server is running before logging)
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("Global_Narrative_Tracker")
        print("MLflow tracking configured.")
    except Exception as e:
        print(f"MLflow initialization skipped: {e}")

# --- Initialize Databases ---
db.init_db()

# --- Initialize LanceDB (Persistent Vector Storage) ---
LANCEDB_URI = "/app/ml_cache/lancedb"
try:
    lance_db = lancedb.connect(LANCEDB_URI)
    try:
        lance_table = lance_db.open_table("articles")
        if "domain" not in lance_table.schema.names:
            lance_db.drop_table("articles")
            raise Exception("Legacy schema detected, recreating table.")
    except Exception:
        # Create schema for 384-dimensional MiniLM embeddings with full metadata
        schema = pa.schema([
            pa.field("doc_id", pa.string()),
            pa.field("url", pa.string()),
            pa.field("title", pa.string()),
            pa.field("domain", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("stance", pa.string()),
            pa.field("confidence", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), 384))
        ])
        lance_table = lance_db.create_table("articles", schema=schema)
    print("LanceDB Vector Storage initialized successfully.")
except Exception as e:
    print(f"Failed to initialize LanceDB: {e}")
    lance_table = None

def reset_lancedb():
    global lance_table, lance_db
    if lance_db is not None:
        try:
            lance_db.drop_table("articles")
        except Exception:
            pass
        schema = pa.schema([
            pa.field("doc_id", pa.string()),
            pa.field("url", pa.string()),
            pa.field("title", pa.string()),
            pa.field("domain", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("stance", pa.string()),
            pa.field("confidence", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), 384))
        ])
        lance_table = lance_db.create_table("articles", schema=schema)

# --- Try loading the Cython LSH module (In-Memory Fast Clustering) ---
cython_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cython_modules")
sys.path.append(cython_dir)

try:
    from lsh import MinHashLSH
    lsh_engine = MinHashLSH(vector_dim=384, num_permutations=128, num_bands=32)
    LSH_LOADED = True
    print("Cython LSH Engine loaded successfully.")
except ImportError:
    print("Warning: Cython LSH Engine not built. Auto-compiling now...")
    import subprocess
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=cython_dir)
    try:
        from lsh import MinHashLSH
        lsh_engine = MinHashLSH(vector_dim=384, num_permutations=128, num_bands=32)
        LSH_LOADED = True
        print("Cython LSH Engine compiled and loaded successfully!")
    except ImportError as e:
        LSH_LOADED = False
        print(f"Failed to auto-compile Cython LSH Engine: {e}")

from fastapi.middleware.cors import CORSMiddleware

# In-memory store for exact verification of LSH candidates
doc_embeddings = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    topic: str
    action: str = "new"
    max_results: int = 15

@app.get("/")
def health_check():
    return {"status": "ok", "lsh_loaded": LSH_LOADED}

@app.get("/api/feed")
def get_feed():
    articles = db.get_latest_articles(limit=50)
    return {"articles": articles}

@app.post("/api/search")
def run_search_pipeline(req: SearchRequest):
    global lsh_engine, doc_embeddings
    
    start_time = time.time()
    
    # MLOps: Start an MLflow Run for this search
    try:
        mlflow.start_run(run_name=f"Search: {req.topic}")
        mlflow.log_param("topic", req.topic)
        mlflow.log_param("action", req.action)
        mlflow.log_param("max_results", req.max_results)
    except:
        pass
    
    if req.action == "new":
        db.reset_db()
        db.init_db()
        doc_embeddings.clear()
        if LSH_LOADED:
            lsh_engine = MinHashLSH(vector_dim=384, num_permutations=128, num_bands=32)
            
    scraped_data = search_and_scrape(req.topic, max_results=req.max_results)
    
    processed_count = 0
    new_clusters_created = 0
    
    for item in scraped_data:
        doc_id = hashlib.md5(item['url'].encode()).hexdigest()
        
        summary = summarize_text(item['full_text'])
        stance_result = get_stance(summary, req.topic)
        
        # Embed text
        emb = get_embedding(item['full_text'])
        
        # MLOps: Store vector in LanceDB persistently
        if lance_table is not None:
            try:
                lance_table.add([{
                    "doc_id": doc_id, 
                    "url": item['url'], 
                    "title": item['title'], 
                    "domain": item['domain'],
                    "summary": summary,
                    "stance": stance_result['stance'],
                    "confidence": float(stance_result['confidence']),
                    "vector": emb.tolist()
                }])
            except Exception as e:
                print(f"LanceDB insertion failed: {e}")
        
        cluster_id = None
        if LSH_LOADED:
            candidates = lsh_engine.query(emb)
            if candidates:
                for cand_id in candidates:
                    cand_emb = doc_embeddings.get(cand_id)
                    if cand_emb is not None:
                        sim = np.dot(emb, cand_emb) / (np.linalg.norm(emb) * np.linalg.norm(cand_emb))
                        if sim > 0.60:
                            cluster_id = db.get_cluster_id(cand_id)
                            break
            
            if not cluster_id:
                cluster_id = str(uuid.uuid4())[:8]
                new_clusters_created += 1
                
            lsh_engine.insert(doc_id, emb)
            doc_embeddings[doc_id] = emb
        else:
            cluster_id = "default"
            
        db.insert_article(
            doc_id=doc_id, 
            url=item['url'], 
            title=item['title'], 
            domain=item['domain'], 
            summary=summary, 
            stance=stance_result['stance'], 
            confidence=stance_result['confidence'],
            cluster_id=cluster_id
        )
        processed_count += 1
        
        # Update custom Prometheus GPU metrics
        update_gpu_metrics()
        
    execution_time = time.time() - start_time
    
    # MLOps: Log final metrics to MLflow and end run
    try:
        mlflow.log_metric("processed_articles", processed_count)
        mlflow.log_metric("new_clusters", new_clusters_created)
        mlflow.log_metric("execution_time_seconds", execution_time)
        mlflow.end_run()
    except:
        pass
        
    return {"status": "success", "processed": processed_count, "topic": req.topic}

@app.post("/api/search_offline")
def search_offline(req: SearchRequest):
    global lsh_engine, doc_embeddings, lance_table
    if lance_table is None:
        raise HTTPException(status_code=500, detail="LanceDB is not active.")
        
    start_time = time.time()
    
    if req.action == "new":
        db.reset_db()
        db.init_db()
        doc_embeddings.clear()
        if LSH_LOADED:
            lsh_engine = MinHashLSH(vector_dim=384, num_permutations=128, num_bands=32)
            
    query_emb = get_embedding(req.topic)
    
    try:
        results = lance_table.search(query_emb.tolist()).limit(req.max_results).to_arrow().to_pylist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LanceDB search failed: {e}")
        
    processed_count = 0
    new_clusters_created = 0
    
    for row in results:
        emb = np.array(row['vector'], dtype=np.float32)
        
        # Enforce a strict Semantic Similarity Threshold
        # Because LanceDB always returns top K results, we must filter out weak matches
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        if sim < 0.35:
            continue
            
        doc_id = row['doc_id']
        url = row['url']
        title = row['title']
        domain = row['domain']
        summary = row['summary']
        stance = row['stance']
        confidence = float(row['confidence'])
        
        cluster_id = None
        if LSH_LOADED:
            candidates = lsh_engine.query(emb)
            if candidates:
                for cand_id in candidates:
                    cand_emb = doc_embeddings.get(cand_id)
                    if cand_emb is not None:
                        sim = np.dot(emb, cand_emb) / (np.linalg.norm(emb) * np.linalg.norm(cand_emb))
                        if sim > 0.60:
                            cluster_id = db.get_cluster_id(cand_id)
                            break
            
            if not cluster_id:
                cluster_id = str(uuid.uuid4())[:8]
                new_clusters_created += 1
                
            lsh_engine.insert(doc_id, emb)
            doc_embeddings[doc_id] = emb
        else:
            cluster_id = "default"
            
        db.insert_article(
            doc_id=doc_id, 
            url=url, 
            title=title, 
            domain=domain, 
            summary=summary, 
            stance=stance, 
            confidence=confidence,
            cluster_id=cluster_id
        )
        processed_count += 1
        
    execution_time = time.time() - start_time
    return {"status": "success", "processed": processed_count, "topic": req.topic, "time": execution_time}

@app.post("/cluster/query")
def query_cluster(req: SearchRequest):
    if not LSH_LOADED:
        raise HTTPException(status_code=501, detail="LSH engine not available")
    embedding = get_embedding(req.topic)
    candidates = lsh_engine.query(embedding)
    return {"candidates": candidates}

@app.post("/api/clear")
def clear_database():
    global lsh_engine, doc_embeddings
    db.reset_db()
    db.init_db()
    doc_embeddings.clear()
    if LSH_LOADED:
        lsh_engine = MinHashLSH(vector_dim=384, num_permutations=128, num_bands=32)
    return {"status": "success", "message": "Database and LSH cache cleared."}

@app.get("/api/lancedb/stats")
def get_lancedb_stats():
    count = 0
    if lance_table is not None:
        try:
            count = len(lance_table)
        except Exception:
            try:
                count = lance_table.count_rows()
            except Exception:
                pass
    return {"count": count}

@app.post("/api/lancedb/clear")
def clear_lancedb():
    reset_lancedb()
    return {"status": "success", "message": "LanceDB cleared."}
