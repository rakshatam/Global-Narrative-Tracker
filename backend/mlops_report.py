import sqlite3
import pandas as pd
import os
import requests
import plotly.express as px

DB_PATH = os.path.join(os.path.dirname(__file__), "tracker.db")

def generate_mlops_report():
    print("Generating MLOps Observability Dashboard (Custom Plotly)...")
    
    # 1. Load Data from SQLite
    if not os.path.exists(DB_PATH):
        print("Database not found. Run ingestion first!")
        return
        
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()
    
    if df.empty:
        print("Database is empty. Run ingestion first!")
        return

    print(f"Loaded {len(df)} articles from the database.")
    
    # 2. Build Custom Interactive Dashboard
    # 1. Stance Distribution
    stance_counts = df['stance'].value_counts().reset_index()
    stance_counts.columns = ['Stance', 'Count']
    fig1 = px.pie(stance_counts, values='Count', names='Stance', title='Overall Narrative Stance Distribution', hole=0.4, 
                  color='Stance', color_discrete_map={'positive':'#00cc96', 'negative':'#ef553b', 'neutral':'#636efa'})
    fig1.update_layout(template="plotly_dark")
    
    # 2. Confidence Distribution
    fig2 = px.histogram(df, x='confidence', nbins=20, title='Model Confidence Distribution')
    fig2.update_layout(template="plotly_dark")
    
    # 3. Stance by Domain
    domain_stance = df.groupby(['domain', 'stance']).size().reset_index(name='count')
    fig3 = px.bar(domain_stance, x='domain', y='count', color='stance', title='Narrative Framing by News Domain', barmode='group',
                  color_discrete_map={'positive':'#00cc96', 'negative':'#ef553b', 'neutral':'#636efa'})
    fig3.update_layout(template="plotly_dark")
    
    # Combine into a single HTML file
    html_content = f"""
    <html>
    <head>
        <title>MLOps Observability Dashboard</title>
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #111111; color: #ffffff; padding: 20px; }}
            h1 {{ text-align: center; border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 40px; font-weight: 300; letter-spacing: 2px; }}
            .chart-container {{ width: 80%; margin: 0 auto 40px auto; border: 1px solid #333; border-radius: 10px; overflow: hidden; }}
        </style>
    </head>
    <body>
        <h1>GLOBAL NARRATIVE MLOPS DASHBOARD</h1>
        <div class="chart-container">{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div class="chart-container">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="chart-container">{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>
    </body>
    </html>
    """
    
    report_path = os.path.join(os.path.dirname(__file__), "mlops_dashboard.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n[SUCCESS] Custom MLOps HTML Dashboard generated at:\n -> {report_path}")
    print("Open this file in your browser to see the interactive ML metrics!")

def verify_lsh():
    """
    Verifies if the Cython LSH engine is actually clustering data.
    """
    print("\n--- Verifying Cython LSH Cluster ---")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM articles LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("No articles in DB to test LSH.")
        return
        
    test_title = row[0]
    print(f"Querying LSH Cluster for near-duplicates to:\n'{test_title}'")
    
    try:
        res = requests.post("http://localhost:8000/cluster/query", json={"doc_id": "test_verification", "text": test_title})
        if res.status_code == 200:
            candidates = res.json().get("candidates", [])
            print(f"\nLSH returned {len(candidates)} candidate URL(s) from the $O(1)$ hash buckets.")
            
            if len(candidates) > 0:
                print(" -> [SUCCESS] LSH is NOT silently failing. It is actively hashing and retrieving documents!")
                for c in candidates:
                    print(f"    - Match: {c}")
            else:
                print(" -> [WARNING] LSH buckets are empty. The in-memory LSH resets if the FastAPI server restarts. Run ingestion again while the server is alive to fill it up.")
        else:
            print(f"LSH Query failed: {res.text}")
    except Exception as e:
        print(f"Could not connect to local API: {e}")

if __name__ == "__main__":
    generate_mlops_report()
    verify_lsh()
