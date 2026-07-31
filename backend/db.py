import sqlite3
import os
import datetime

# Store the database file in the same directory as this script
DB_PATH = os.path.join(os.path.dirname(__file__), "tracker.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_latest_articles(limit=50):
    """
    Fetches the latest articles for the frontend dashboard.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT doc_id, url, title, domain, summary, stance, confidence, cluster_id, timestamp 
        FROM articles 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert rows to a list of dicts
    return [dict(row) for row in rows]

def reset_db():
    """
    Wipes the articles table. Used when the user starts a completely new topic search
    so the old cluster is destroyed to save disk space and RAM.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM articles')
    conn.commit()
    cursor.execute('VACUUM') # Reclaim disk space immediately
    conn.commit()
    conn.close()
    print("[DB] Database wiped for new search cluster.")

def init_db():
    """Initializes the SQLite database and creates the necessary tables."""
    conn = get_connection()
    c = conn.cursor()
    
    # We are dropping the old table because we changed the schema to add url, summary, and cluster_id
    c.execute('DROP TABLE IF EXISTS articles')
    
    # Create the articles table
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE,
            url TEXT,
            title TEXT,
            domain TEXT,
            summary TEXT,
            stance TEXT,
            confidence REAL,
            cluster_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Create an index on timestamp to make the automatic pruning extremely fast
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON articles(timestamp)')
    conn.commit()
    conn.close()
    print(f"SQLite database initialized at {DB_PATH}")

def insert_article(doc_id: str, url: str, title: str, domain: str, summary: str, stance: str, confidence: float, cluster_id: str):
    """Inserts a processed article into the database."""
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO articles (doc_id, url, title, domain, summary, stance, confidence, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (doc_id, url, title, domain, summary, stance, confidence, cluster_id))
        conn.commit()
    except sqlite3.IntegrityError:
        # If we fetch the same article twice, SQLite will safely ignore the duplicate
        pass
    finally:
        conn.close()

def get_cluster_id(doc_id: str) -> str:
    """Returns the cluster_id for a given doc_id, or None if not found."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT cluster_id FROM articles WHERE doc_id = ?', (doc_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def prune_old_articles(hours_to_keep=72):
    """
    Deletes articles older than the specified hours to save disk space.
    Runs a VACUUM command to physically release the disk space back to the OS.
    """
    conn = get_connection()
    c = conn.cursor()
    # SQLite CURRENT_TIMESTAMP is in UTC
    cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(hours=hours_to_keep)
    
    c.execute('DELETE FROM articles WHERE timestamp < ?', (cutoff_time,))
    deleted_count = c.rowcount
    conn.commit()
    
    # If we deleted rows, run VACUUM to reclaim the actual disk bytes
    if deleted_count > 0:
        c.execute('VACUUM')
        
    conn.close()
    return deleted_count
