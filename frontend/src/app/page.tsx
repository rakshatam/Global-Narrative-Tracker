"use client";

import { useState, useEffect } from "react";
import { Search, Loader2, Activity, ShieldAlert, BarChart3, Globe, DatabaseZap, Clock } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from "recharts";

type Article = {
  doc_id: string;
  url: string;
  title: string;
  domain: string;
  summary: string;
  stance: string;
  confidence: number;
  cluster_id: string;
  timestamp: string;
};

export default function Dashboard() {
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [feed, setFeed] = useState<Article[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lanceCount, setLanceCount] = useState<number>(0);

  // Polling for live feed updates
  const fetchFeed = async () => {
    try {
      const res = await fetch("/api/feed");
      const data = await res.json();
      if (data.articles) setFeed(data.articles);
      
      const resStats = await fetch("/api/lancedb/stats");
      const dataStats = await resStats.json();
      if (dataStats.count !== undefined) setLanceCount(dataStats.count);
    } catch (err) {
      console.error("Failed to fetch feed", err);
    }
  };

  useEffect(() => {
    fetchFeed();
    const interval = setInterval(fetchFeed, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleSearch = async (e: React.FormEvent, action: "new" | "update" = "new") => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setIsSearching(true);
    setError(null);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: query, action, max_results: 15 })
      });
      const data = await res.json();
      if (data.status !== "success") {
        setError(data.detail || "Search failed.");
      }
      await fetchFeed();
    } catch (err) {
      setError("Network error connecting to ML Engine.");
    } finally {
      setIsSearching(false);
    }
  };

  const handleOfflineSearch = async () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setError(null);
    try {
      const res = await fetch("/api/search_offline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: query, action: "new", max_results: 20 })
      });
      const data = await res.json();
      if (data.status !== "success") {
        setError(data.detail || "Offline Search failed.");
      }
      await fetchFeed();
    } catch (err) {
      setError("Network error connecting to ML Engine.");
    } finally {
      setIsSearching(false);
    }
  };

  // Prepare data for charts
  const stanceCounts = feed.reduce((acc, curr) => {
    acc[curr.stance] = (acc[curr.stance] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const chartData = [
    { name: "Positive", value: stanceCounts["positive"] || 0, color: "#10b981" },
    { name: "Negative", value: stanceCounts["negative"] || 0, color: "#ef4444" },
    { name: "Neutral", value: stanceCounts["neutral"] || 0, color: "#6b7280" }
  ].filter(d => d.value > 0);

  const handleClearDB = async () => {
    try {
      await fetch("/api/clear", { method: "POST" });
      setFeed([]);
    } catch (err) {
      console.error(err);
    }
  };

  const handleClearLanceDB = async () => {
    try {
      await fetch("/api/lancedb/clear", { method: "POST" });
      setLanceCount(0);
    } catch (err) {
      console.error(err);
    }
  };

  const groupedFeed = feed.reduce((acc, curr) => {
    const cluster = curr.cluster_id || "unclustered";
    if (!acc[cluster]) acc[cluster] = [];
    acc[cluster].push(curr);
    return acc;
  }, {} as Record<string, Article[]>);

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-white font-sans p-6">
      
      {/* HEADER & SEARCH */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center mb-8 gap-4 border-b border-gray-800 pb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-500/10 rounded-xl border border-blue-500/20">
            <Activity className="text-blue-400 w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-gray-100">Global Narrative Tracker</h1>
            <p className="text-sm text-gray-400">Live AI Stance Detection & LSH Clustering</p>
          </div>
        </div>

        <form onSubmit={(e) => handleSearch(e, "new")} className="relative w-full md:w-96 flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input 
              type="text" 
              placeholder="Enter a topic (e.g. US Economy)..."
              className="w-full bg-gray-900/50 border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:border-blue-500 transition-colors"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <button 
            type="submit" 
            disabled={isSearching}
            className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center gap-2 whitespace-nowrap"
            title="Live Web Scraping Search"
          >
            {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : "Live Search"}
          </button>
          <button 
            type="button" 
            onClick={handleOfflineSearch}
            disabled={isSearching}
            className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center gap-2 whitespace-nowrap"
            title="Search Historical Offline Memory"
          >
            {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : "Offline Search"}
          </button>
        </form>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* LEFT COLUMN: LIVE FEED */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Globe className="w-5 h-5 text-gray-400" />
              Live Narrative Feed
            </h2>
            <div className="flex items-center gap-2 text-xs font-medium text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-full">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              Engine Active
            </div>
          </div>

          <div className="flex-1 bg-gray-900/40 border border-gray-800 rounded-xl p-4 overflow-y-auto h-[700px] flex flex-col gap-4">
            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg flex items-center gap-2">
                <ShieldAlert className="w-5 h-5" />
                {error}
              </div>
            )}
            
            {feed.length === 0 && !isSearching && (
              <div className="h-full flex flex-col items-center justify-center text-gray-500 gap-2">
                <DatabaseZap className="w-12 h-12 opacity-20" />
                <p>No narratives tracked yet. Search for a topic to begin.</p>
              </div>
            )}

            {Object.entries(groupedFeed).map(([clusterId, clusterArticles]) => (
              <div key={clusterId} className="mb-8 last:mb-0">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-md text-xs font-bold border border-purple-500/30 tracking-wider">
                    NARRATIVE CLUSTER: {clusterId.toUpperCase()}
                  </div>
                  <div className="h-px flex-1 bg-gradient-to-r from-purple-500/20 to-transparent"></div>
                  <span className="text-xs text-gray-500 font-medium bg-black/40 px-2 py-1 rounded-full border border-white/5">
                    {clusterArticles.length} related articles
                  </span>
                </div>
                
                <div className="flex flex-col gap-4">
                  {clusterArticles.map((article) => (
                    <a key={article.doc_id} href={article.url} target="_blank" rel="noreferrer" className="block group">
                      <div className="bg-gray-900 border border-gray-800 hover:border-gray-700 rounded-lg p-5 transition-all hover:bg-gray-800/50 relative overflow-hidden">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-gray-800 group-hover:bg-purple-500/50 transition-colors"></div>
                        <div className="flex items-start justify-between gap-4 mb-3 pl-2">
                          <h3 className="text-md font-medium text-gray-200 group-hover:text-blue-400 transition-colors leading-snug">
                            {article.title}
                          </h3>
                          <span className={`px-2.5 py-1 rounded-md text-xs font-bold whitespace-nowrap ${
                            article.stance === "positive" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" :
                            article.stance === "negative" ? "bg-red-500/10 text-red-400 border border-red-500/20" :
                            "bg-gray-500/10 text-gray-400 border border-gray-500/20"
                          }`}>
                            {article.stance.toUpperCase()} ({(article.confidence * 100).toFixed(0)}%)
                          </span>
                        </div>
                        <p className="text-sm text-gray-400 leading-relaxed mb-4 line-clamp-3 pl-2">
                          {article.summary}
                        </p>
                        <div className="flex items-center gap-4 text-xs text-gray-500 font-medium pl-2">
                          <span className="flex items-center gap-1.5 bg-black/40 px-2 py-1 rounded-md border border-white/5">
                            <Globe className="w-3.5 h-3.5" />
                            {article.domain}
                          </span>
                          <span className="flex items-center gap-1.5">
                            <Clock className="w-3.5 h-3.5" />
                            {new Date(article.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT COLUMN: ANALYTICS */}
        <div className="flex flex-col gap-6">
          <div className="bg-gray-900/40 border border-gray-800 rounded-xl p-5">
            <h2 className="text-sm font-semibold flex items-center gap-2 text-gray-300 mb-6 uppercase tracking-wider">
              <BarChart3 className="w-4 h-4 text-purple-400" />
              Stance Distribution
            </h2>
            <div className="h-64 w-full">
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#111', borderColor: '#333', borderRadius: '8px' }}
                      itemStyle={{ color: '#fff' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                  No data to visualize
                </div>
              )}
            </div>
            
            <div className="flex justify-center gap-4 mt-2">
              {chartData.map(d => (
                <div key={d.name} className="flex items-center gap-2 text-xs text-gray-400">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }}></div>
                  {d.name} ({d.value})
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-900/30 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-blue-300 mb-2">LSH Clustering Engine</h3>
            <p className="text-xs text-gray-400 leading-relaxed mb-4">
              Cython-accelerated MinHash LSH is currently hashing summaries into high-dimensional buckets. 
              Search a new topic to instantly purge the database and reset the cluster.
            </p>
            <div className="flex justify-between items-center bg-black/40 rounded-lg p-3 border border-white/5">
              <span className="text-xs text-gray-400">Memory Usage</span>
              <span className="text-xs font-mono text-emerald-400">14.2 MB</span>
            </div>
            <div className="flex justify-between items-center bg-black/40 rounded-lg p-3 border border-white/5 mt-2">
              <span className="text-xs text-gray-400">LSH Band/Row config</span>
              <span className="text-xs font-mono text-emerald-400">32 / 4</span>
            </div>
            <div className="flex justify-between items-center bg-black/40 rounded-lg p-3 border border-white/5 mt-2">
              <span className="text-xs text-gray-400">LanceDB Vectors</span>
              <span className="text-xs font-mono text-purple-400">{lanceCount} saved</span>
            </div>
            
            <button 
              onClick={handleClearDB}
              className="w-full mt-4 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 hover:border-red-500/40 py-2 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2"
            >
              <DatabaseZap className="w-4 h-4" />
              Purge Database & Clusters
            </button>
            <button 
              onClick={handleClearLanceDB}
              className="w-full mt-2 bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 border border-purple-500/20 hover:border-purple-500/40 py-2 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2"
            >
              <DatabaseZap className="w-4 h-4" />
              Flush LanceDB Vectors
            </button>
          </div>
        </div>

      </main>
    </div>
  );
}
