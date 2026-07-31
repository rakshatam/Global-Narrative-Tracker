import time
import requests
import schedule
import json
import os
import shutil
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

# Configuration
# Using the GDELT 2.0 DOC API for easy JSON fetching
# We search for a broad geopolitical topic to see how different outlets frame it.
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
LOCAL_API_BASE = "http://localhost:8000"
TOPIC = "Global Economy"
QUERY_PARAM = "economy" # Search query for GDELT
MAX_DISK_GB = 15.0

def check_disk_failsafe():
    """
    Emergency failsafe: Checks if the disk hosting the project has exceeded 
    the strict 15GB limit to protect the user's local machine.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Get total size of the project directory
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(project_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
                
    size_gb = total_size / (1024 * 1024 * 1024)
    
    if size_gb > MAX_DISK_GB:
        print(f"CRITICAL: Project size ({size_gb:.2f} GB) exceeded 15GB limit. Halting ingestion to protect disk.")
        exit(1)
        
    print(f"[Disk Check] Project footprint is safe: {size_gb:.2f} / {MAX_DISK_GB} GB used.")

def fetch_latest_news():
    """
    Fetches the latest global news articles directly from public RSS feeds.
    This completely bypasses GDELT and any 3rd party API rate limits (429 errors).
    """
    print(f"Fetching latest news from global RSS firehose...")
    
    rss_feeds = [
        "http://feeds.bbci.co.uk/news/world/rss.xml",       # BBC World
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml",      # Wall Street Journal
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml" # NY Times
    ]
    
    # Use a standard browser User-Agent so news sites don't block us
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    articles = []
    
    for feed_url in rss_feeds:
        try:
            # Extract domain name for our database (e.g., 'bbci.co.uk')
            domain = urlparse(feed_url).netloc.replace("feeds.", "").replace("rss.", "")
            
            response = requests.get(feed_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Grab the top 5 most recent articles from each feed
            for item in root.findall('.//item')[:5]: 
                title_elem = item.find('title')
                link_elem = item.find('link')
                
                if title_elem is not None and link_elem is not None:
                    articles.append({
                        "url": link_elem.text,
                        "title": title_elem.text,
                        "domain": domain
                    })
        except Exception as e:
            print(f"Warning: Failed to fetch from {feed_url} - {e}")
            
    print(f"Successfully fetched {len(articles)} real-world articles from RSS firehose.\n")
    return articles

def process_article(article):
    """Sends a single article through our local ML pipeline."""
    url = article.get("url", "unknown_url")
    title = article.get("title", "")
    domain = article.get("domain", "unknown_domain")
    
    if not title:
        return
        
    print(f"\n--- Processing Article from {domain} ---")
    print(f"Title: {title}")
    
    # 1. Insert into Cython LSH Cluster
    # We use the URL as a unique doc_id
    cluster_payload = {
        "doc_id": url,
        "text": title
    }
    try:
        res = requests.post(f"{LOCAL_API_BASE}/cluster/insert", json=cluster_payload)
        if res.status_code == 200:
            print(" -> [LSH] Inserted into cluster successfully.")
        else:
            print(f" -> [LSH] Warning: {res.text}")
    except requests.exceptions.ConnectionError:
        print(" -> [Error] Local API is offline. Is the FastAPI server running?")
        return

    # 2. Analyze Stance using Zero-Shot Classifier and save to SQLite
    stance_payload = {
        "topic": TOPIC,
        "text": title,
        "doc_id": url,
        "domain": domain
    }
    try:
        res = requests.post(f"{LOCAL_API_BASE}/analyze/stance", json=stance_payload)
        if res.status_code == 200:
            stance_data = res.json()
            stance = stance_data.get('stance', 'unknown')
            confidence = stance_data.get('confidence', 0.0)
            print(f" -> [ML Stance] Prediction: {stance.upper()} (Confidence: {confidence:.2f})")
        else:
            print(f" -> [ML Stance] Error: {res.text}")
    except Exception as e:
         print(f" -> [ML Stance] Failed to analyze: {e}")

def test_run():
    """Runs a single test iteration of the ingestion and processing pipeline."""
    check_disk_failsafe()
    articles = fetch_latest_news()
    for article in articles:
        process_article(article)
        time.sleep(0.5) # Slight delay to not overwhelm the local API during testing

def job():
    """The recurring job for production."""
    check_disk_failsafe()
    articles = fetch_latest_news()
    for article in articles:
        process_article(article)
        
    # Prune old articles from SQLite to maintain the rolling window (e.g. 72 hours)
    import db
    deleted = db.prune_old_articles(hours_to_keep=72)
    if deleted > 0:
        print(f"[Maintenance] Pruned {deleted} old articles from SQLite to save disk space.")

if __name__ == "__main__":
    print("Starting GDELT Ingestion Test Run...")
    test_run()
    
    # For production background worker, uncomment below:
    # print("Starting continuous ingestion scheduler (runs every 15 minutes)...")
    # schedule.every(15).minutes.do(job)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
