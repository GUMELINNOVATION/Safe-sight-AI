"""
CSV to HTML Records Converter
==============================
Run this script after generating worker_safety_report.csv to create
a standalone records.html file with all data embedded.

Usage:
    python generate_records.py

This creates records.html with all CSV data embedded as JSON.
No server needed - just open the file in any browser.
"""

import csv
import json
import os
from datetime import datetime

CSV_PATH = "worker_safety_report.csv"
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise PPE Compliance | Records HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-bg: #030712;
            --glass-bg: rgba(17, 24, 39, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --safe: #10b981;
            --warning: #f59e0b;
            --critical: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --sidebar-w: 320px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--primary-bg);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            background-image: 
                radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 100% 100%, rgba(16, 185, 129, 0.1) 0%, transparent 40%);
        }

        /* Layout */
        .app-shell {
            display: grid;
            grid-template-columns: var(--sidebar-w) 1fr;
            height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
            z-index: 50;
        }

        .sidebar-header {
            padding: 2rem 1.5rem;
            border-bottom: 1px solid var(--glass-border);
        }

        .brand h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
        }

        .search-container {
            position: relative;
        }

        .search-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.9rem;
            outline: none;
            transition: all 0.3s;
        }

        .search-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .worker-list {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }

        .worker-item {
            padding: 1rem;
            border-radius: 16px;
            margin-bottom: 0.75rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 1rem;
            border: 1px solid transparent;
        }

        .worker-item:hover {
            background: rgba(255, 255, 255, 0.03);
            transform: translateX(4px);
        }

        .worker-item.active {
            background: rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        .worker-avatar {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            border: 1px solid var(--glass-border);
        }

        .worker-info h4 { font-size: 0.95rem; margin-bottom: 0.2rem; }
        .worker-info p { font-size: 0.75rem; color: var(--text-muted); }

        .worker-badge {
            margin-left: auto;
            font-size: 0.65rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            text-transform: uppercase;
        }

        .badge-safe { color: var(--safe); background: rgba(16, 185, 129, 0.1); }
        .badge-critical { color: var(--critical); background: rgba(239, 68, 68, 0.1); }

        /* Main Content */
        .main-view {
            padding: 2.5rem;
            overflow-y: auto;
            position: relative;
        }

        .top-stats {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .summary-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            padding: 1.5rem;
            border-radius: 24px;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .summary-card span { font-size: 0.8rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
        .summary-card h2 { font-family: 'Outfit', sans-serif; font-size: 2rem; color: #fff; }

        /* Detail View */
        .profile-hero {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.5));
            border: 1px solid var(--glass-border);
            border-radius: 32px;
            padding: 3rem;
            margin-bottom: 2.5rem;
            display: flex;
            align-items: center;
            gap: 3rem;
            position: relative;
            overflow: hidden;
        }

        .profile-hero::after {
            content: "";
            position: absolute;
            top: -50%;
            right: -10%;
            width: 400px;
            height: 400px;
            background: var(--accent);
            filter: blur(120px);
            opacity: 0.1;
        }

        .hero-avatar {
            width: 120px;
            height: 120px;
            background: var(--primary-bg);
            border-radius: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 4rem;
            border: 2px solid var(--glass-border);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }

        .hero-text h2 { font-family: 'Outfit', sans-serif; font-size: 2.5rem; margin-bottom: 0.5rem; }
        .id-badge { background: rgba(255, 255, 255, 0.05); padding: 0.4rem 1rem; border-radius: 12px; font-size: 0.85rem; color: var(--text-muted); border: 1px solid var(--glass-border); }

        .stat-grid {
            margin-left: auto;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        .stat-item { text-align: center; background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 20px; border: 1px solid var(--glass-border); min-width: 120px; }
        .stat-val { display: block; font-size: 1.75rem; font-weight: 700; font-family: 'Outfit', sans-serif; }
        .stat-lab { font-size: 0.7rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; }

        /* Tables */
        .record-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 32px;
            padding: 2rem;
            box-shadow: 0 30px 60px rgba(0,0,0,0.3);
        }

        .record-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }

        .filter-tabs { display: flex; gap: 0.75rem; background: rgba(0,0,0,0.3); padding: 0.5rem; border-radius: 16px; border: 1px solid var(--glass-border); }
        .filter-tab { background: transparent; border: none; color: var(--text-muted); padding: 0.6rem 1.5rem; border-radius: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-size: 0.85rem; }
        .filter-tab.active { background: var(--accent); color: #fff; box-shadow: 0 4px 15px var(--accent-glow); }

        .export-btn {
            background: linear-gradient(to right, #1e293b, #0f172a);
            border: 1px solid var(--glass-border);
            color: #fff;
            padding: 0.8rem 1.8rem;
            border-radius: 14px;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .export-btn:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,0.3); }

        table { width: 100%; border-collapse: separate; border-spacing: 0 12px; }
        th { text-align: left; padding: 0 1.5rem; font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; }
        td { padding: 1.5rem; background: rgba(255, 255, 255, 0.02); border-top: 1px solid var(--glass-border); border-bottom: 1px solid var(--glass-border); }
        td:first-child { border-left: 1px solid var(--glass-border); border-radius: 16px 0 0 16px; }
        td:last-child { border-right: 1px solid var(--glass-border); border-radius: 0 16px 16px 0; }

        .timestamp { font-family: monospace; color: var(--text-muted); font-size: 0.85rem; }
        .status-badge { display: inline-flex; align-items: center; gap: 0.6rem; padding: 0.5rem 1rem; border-radius: 12px; font-weight: 700; font-size: 0.75rem; }
        .bg-safe { background: rgba(16, 185, 129, 0.1); color: var(--safe); }
        .bg-critical { background: rgba(239, 68, 68, 0.1); color: var(--critical); }
        .dot-lite { width: 6px; height: 6px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }

        .empty-view { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh; opacity: 0.3; }
        .empty-view h1 { font-size: 6rem; margin-bottom: 1rem; }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="app-shell">
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="brand">
                    <h1>SAFE-SIGHT HUD</h1>
                </div>
                <div class="search-container">
                    <input type="text" id="workerSearch" class="search-input" placeholder="Search safety registry...">
                </div>
            </div>
            <ul class="worker-list" id="workerList"></ul>
        </aside>

        <main class="main-view" id="mainContent">
            <div class="top-stats">
                <div class="summary-card">
                    <span>Registry Size</span>
                    <h2 id="total-workers">0</h2>
                </div>
                <div class="summary-card">
                    <span>Safety Audit Logs</span>
                    <h2 id="total-records">0</h2>
                </div>
                <div class="summary-card">
                    <span>Active Violations</span>
                    <h2 id="total-violations" style="color:var(--critical)">0</h2>
                </div>
            </div>

            <div id="workerDetail" class="empty-view">
                <h1>🛡️</h1>
                <h3>Registry Locked</h3>
                <p>Select a worker to decrypt safety history</p>
            </div>
        </main>
    </div>

    <script>
        const EMBEDDED_DATA = {{EMBEDDED_JSON}};

        let allRecords = EMBEDDED_DATA.records || [];
        let workers = {};
        let currentWorker = null;
        let currentFilter = 'all';

        function processWorkers() {
            workers = {};
            allRecords.forEach(r => {
                const key = `${r.Face_ID || 'Unknown'}-${r.Name || 'Unknown'}`;
                if (!workers[key]) {
                    workers[key] = { 
                        faceId: r.Face_ID || 'Unknown', 
                        name: r.Name || 'Unknown Worker',
                        records: [], violations: 0, safeCount: 0, lastSeen: null 
                    };
                }
                workers[key].records.push(r);
                if (r.Event === 'VIOLATION') workers[key].violations++;
                else workers[key].safeCount++;
                const t = new Date(r.Timestamp);
                if (!workers[key].lastSeen || t > workers[key].lastSeen) workers[key].lastSeen = t;
            });
        }

        function renderWorkerList(term = '') {
            const list = document.getElementById('workerList');
            const filtered = Object.entries(workers).filter(([k, w]) => {
                const t = term.toLowerCase();
                return w.name.toLowerCase().includes(t) || w.faceId.toString().toLowerCase().includes(t);
            });
            
            list.innerHTML = filtered.map(([k, w]) => {
                const active = currentWorker === k;
                const badgeClass = w.violations > 0 ? 'badge-critical' : 'badge-safe';
                return `
                <li class="worker-item ${active ? 'active' : ''}" onclick="selectWorker('${k}')" data-key="${k}">
                    <div class="worker-avatar">👷</div>
                    <div class="worker-info">
                        <h4>${esc(w.name)}</h4>
                        <p>ID: ${w.faceId}</p>
                    </div>
                    <span class="worker-badge ${badgeClass}">${w.violations > 0 ? 'Issue' : 'Safe'}</span>
                </li>`;
            }).join('');
        }

        function selectWorker(key) {
            currentWorker = key;
            document.querySelectorAll('.worker-item').forEach(el => el.classList.toggle('active', el.dataset.key === key));
            renderDetail(workers[key]);
        }

        function renderDetail(w) {
            const total = w.records.length;
            const rate = total > 0 ? ((w.violations / total) * 100).toFixed(1) : 0;
            document.getElementById('mainContent').innerHTML = `
                <div class="profile-hero">
                    <div class="hero-avatar">👷</div>
                    <div class="hero-text">
                        <h2>${esc(w.name)}</h2>
                        <span class="id-badge">Registry ID: ${w.faceId}</span>
                    </div>
                    <div class="stat-grid">
                        <div class="stat-item"><span class="stat-val" style="color:var(--safe)">${w.safeCount}</span><span class="stat-lab">Safe</span></div>
                        <div class="stat-item"><span class="stat-val" style="color:var(--warning)">${total}</span><span class="stat-lab">Logs</span></div>
                        <div class="stat-item"><span class="stat-val" style="color:var(--critical)">${w.violations}</span><span class="stat-lab">Violations</span></div>
                        <div class="stat-item"><span class="stat-val">${rate}%</span><span class="stat-lab">Risk</span></div>
                    </div>
                </div>

                <div class="record-card">
                    <div class="record-header">
                        <div class="filter-tabs">
                            <button class="filter-tab ${currentFilter === 'all' ? 'active' : ''}" onclick="setFilter('all')">History</button>
                            <button class="filter-tab ${currentFilter === 'violation' ? 'active' : ''}" onclick="setFilter('violation')">Safety Issues</button>
                        </div>
                        <button class="export-btn" onclick="exportCSV('${w.faceId}')">📥 EXPORT RECORDS</button>
                    </div>
                    <table>
                        <thead><tr><th>Timestamp</th><th>Status</th><th>PPE Audit</th><th>Track</th></tr></thead>
                        <tbody>${renderRecords(w.records)}</tbody>
                    </table>
                </div>`;
        }

        function renderRecords(records) {
            let f = records;
            if (currentFilter === 'violation') f = records.filter(r => r.Event === 'VIOLATION');
            if (f.length === 0) return '<tr><td colspan="4" style="text-align:center; padding:3rem; opacity:0.5;">No records available</td></tr>';
            
            return f.map(r => {
                const bad = r.Event === 'VIOLATION';
                return `<tr>
                    <td class="timestamp">${esc(r.Timestamp)}</td>
                    <td>
                        <span class="status-badge ${bad ? 'bg-critical' : 'bg-safe'}">
                            <span class="dot-lite"></span>${bad ? 'VIOLATION' : 'COMPLIANT'}
                        </span>
                    </td>
                    <td style="font-size:0.8rem">
                        <div style="color:${bad && r.Helmet_Status.includes('No') ? 'var(--critical)' : 'var(--safe)'}">H: ${esc(r.Helmet_Status)}</div>
                        <div style="color:${bad && r.Vest_Status.includes('No') ? 'var(--critical)' : 'var(--safe)'}">V: ${esc(r.Vest_Status)}</div>
                    </td>
                    <td class="timestamp">${esc(r.Track_ID)}</td>
                </tr>`;
            }).join('');
        }

        function setFilter(f) { currentFilter = f; if (currentWorker) renderDetail(workers[currentWorker]); }
        function updateStats() {
            // Check if stats elements exist before updating (they are removed in detail view)
            const tw = document.getElementById('total-workers');
            const tr = document.getElementById('total-records');
            const tv = document.getElementById('total-violations');
            if (tw) tw.textContent = Object.keys(workers).length;
            if (tr) tr.textContent = allRecords.length;
            if (tv) tv.textContent = allRecords.filter(r => r.Event === 'VIOLATION').length;
        }
        function esc(t) { if (!t) return ''; const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
        
        function exportCSV(faceId) {
            const w = workers[currentWorker];
            let csv = 'Timestamp,Face_ID,Track_ID,Name,Event,Helmet_Status,Vest_Status\\n';
            w.records.forEach(r => csv += `${r.Timestamp},${r.Face_ID},${r.Track_ID},${r.Name},${r.Event},${r.Helmet_Status},${r.Vest_Status}\\n`);
            const blob = new Blob([csv], {type: 'text/csv'});
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
            a.download = `worker_${faceId}_safety_report.csv`; a.click();
        }

        document.getElementById('workerSearch').addEventListener('input', e => renderWorkerList(e.target.value));

        processWorkers();
        renderWorkerList();
        updateStats();
    </script>
</body>
</html>
"""


def csv_to_json(csv_path):
    """Convert CSV file to JSON array."""
    records = []
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found. Creating empty records file.")
        return records

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))

    print(f"[OK] Loaded {len(records)} records from {csv_path}")
    return records


def generate_html():
    """Generate standalone HTML with embedded CSV data."""
    records = csv_to_json(CSV_PATH)

    data = {
        "records": records,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(records)
    }

    json_str = json.dumps(data, ensure_ascii=False)
    html_content = HTML_TEMPLATE.replace("{{EMBEDDED_JSON}}", json_str)

    output_path = "records.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[OK] Generated {output_path} with {len(records)} records embedded")
    print(f"[INFO] Open {output_path} in any browser - no server needed!")


if __name__ == "__main__":
    generate_html()
