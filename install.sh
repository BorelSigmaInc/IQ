#!/bin/bash
set -e

INSTALL_DIR="$HOME/quantiq-client"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

python3 -m venv venv
source venv/bin/activate
pip install -q requests matplotlib pandas numpy mplcursors

cat > quantiq_client.py << 'PYEOF'
#!/usr/bin/env python3
"""QuantIQ Client – Full table + interactive parallel coordinates."""

import sys, os, json, webbrowser
import requests
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from pathlib import Path

CONFIG_FILE = Path.home() / ".quantiq_config.json"
API_BASE = "http://46.224.200.113:8001"

def load_key():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f).get("api_key")
    return None

def save_key(key):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"api_key": key}, f)

def validate_file(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File '{filepath}' not found.")
        return False
    valid_ext = {".csv", ".json", ".tsv", ".xlsx"}
    if path.suffix.lower() not in valid_ext:
        print(f"Error: Unsupported file type '{path.suffix}'. Supported: {', '.join(valid_ext)}")
        return False
    return True

def main():
    interactive = sys.stdin.isatty()
    print("=" * 60)
    print("  Q U A N T I Q   L E A D   I N T E L L I G E N C E")
    print("=" * 60)

    key = load_key()
    if not key:
        if interactive:
            key = input("Enter your QuantIQ API key: ").strip()
            save_key(key)
            print("Key saved.")
        else:
            print("No API key found. Run interactively first with:  ./quantiq_client.py")
            return
    else:
        print(f"Using saved API key: {key[:20]}...")

    # Optional own data file
    filename = None
    if interactive:
        print("\nOptional: Provide your own lead file (press Enter to skip).")
        filename = input("Data file name (e.g., my_leads.csv): ").strip()
    if filename:
        if interactive:
            filetype = input("Type of file (csv/json/tsv/xlsx): ").strip().lower()
            folder = input("Path of folder containing the file: ").strip()
            full_path = os.path.join(folder, filename)
            if validate_file(full_path):
                consent = input(f"Allow QuantIQ to access '{full_path}'? (Y/N): ").strip().upper()
                if consent == "Y":
                    print("File accepted. (Upload feature will be added soon – currently using demo leads.)")
                else:
                    print("Consent denied. Using pre-loaded leads.")
            else:
                print("File invalid. Continuing with demo leads.")
        else:
            print("File upload not supported in non-interactive mode. Using demo leads.")
    else:
        if not interactive:
            print("No file provided. Using pre-loaded leads from QuantIQ server.")

    limit = 10
    if interactive:
        ans = input("\nHow many leads would you like to score? (default 10): ").strip()
        if ans.isdigit():
            limit = int(ans)
    else:
        print("Non-interactive run – using default of 10 leads.")

    print(f"\nFetching {limit} leads from QuantIQ API...")
    try:
        resp = requests.get(f"{API_BASE}/api/v1/leads", params={"api_key": key, "limit": limit})
        data = resp.json()
    except Exception as e:
        print(f"Error contacting API: {e}")
        return

    if "leads" not in data:
        print("API error:", data.get("detail", "unknown"))
        return

    leads = data["leads"]
    graphs = data.get("graphs", {})
    usage = data.get("usage", {})

    # ---------- Terminal table ----------
    print(f"\nCall #{usage.get('call_number', '?')}  |  {limit} leads")
    headers = ["Priority", "LeadID", "Intent", "Kernel", "RBF", "Gap", "Unc", "Ent", "QFeat"]
    row_fmt = "{:<10}" * len(headers)
    print(row_fmt.format(*headers))
    print("-" * 90)

    plot_rows = []
    for ld in leads:
        q = ld["score_quantum_kernel"]
        c = ld["score_classical_rbf"]
        gap = round(q - c, 2)
        priority = round(q * (1 + gap), 2)
        unc = ld["quantum_uncertainty"]
        ent = ld["entanglement_proxy"]
        qfeat = ld["q_feature_importance"]
        row = [priority, ld["lead_id"], ld["intent"], q, c, gap, unc, ent, qfeat]
        print(row_fmt.format(*row))
        plot_rows.append(row)

    # Domain constants
    print("\nDomain Performance:")
    print("  Prob   = 0.99")
    print("  LinAlg = 0.87")
    print("  QMech  = 0.62")
    print("  Stats  = 1.0")
    print("  OpsRes = -10")

    # ---------- Interactive parallel coordinates ----------
    if not plot_rows:
        return
    data_arr = np.array(plot_rows)
    axes_idx = [0, 3, 4, 5, 6, 7, 8]
    axes_names = ['Priority','Kernel','RBF','Gap','Unc','Ent','QFeat']
    n_axes = len(axes_idx)
    n_leads = data_arr.shape[0]
    lead_ids = data_arr[:,1].astype(int)
    intent_vals = data_arr[:,2].astype(int)
    priority_vals = data_arr[:,0].astype(float)

    X = np.zeros((n_leads, n_axes))
    for i, col_idx in enumerate(axes_idx):
        col = data_arr[:, col_idx].astype(float)
        if col_idx == 8:
            X[:, i] = col / 15.0
        else:
            minv, maxv = col.min(), col.max()
            X[:, i] = (col - minv) / (maxv - minv) if maxv > minv else 0.5

    fig, ax = plt.subplots(figsize=(12,6))
    lines = []
    for i in range(n_leads):
        cmap = plt.cm.Blues if intent_vals[i] == 1 else plt.cm.Oranges
        c_intensity = 0.2 + 0.8 * (priority_vals[i] - priority_vals.min()) / (priority_vals.max() - priority_vals.min())
        line, = ax.plot(range(n_axes), X[i], marker='o', markersize=2,
                        linewidth=0.8, alpha=0.7, color=cmap(c_intensity))
        lines.append(line)

    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(axes_names)
    ax.set_title('Interactive Parallel Coordinates – Hover to highlight a lead')

    def on_hover(sel):
        idx = lines.index(sel.artist)
        for j, l in enumerate(lines):
            if j == idx:
                l.set_linewidth(2.0)
                l.set_alpha(1.0)
            else:
                l.set_linewidth(0.6)
                l.set_alpha(0.3)
        sel.annotation.set_text(f"Lead ID: {lead_ids[idx]}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
        fig.canvas.draw_idle()

    def on_mouse_move(event):
        if event.inaxes != ax:
            for l in lines:
                l.set_linewidth(0.8)
                l.set_alpha(0.7)
            fig.canvas.draw_idle()
            return
        over = any(line.contains(event)[0] for line in lines)
        if not over:
            for l in lines:
                l.set_linewidth(0.8)
                l.set_alpha(0.7)
            fig.canvas.draw_idle()

    cursor = mplcursors.cursor(lines, hover=True)
    cursor.connect("add", on_hover)
    cursor.connect("remove", lambda sel: None)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    plt.tight_layout()
    plt.show()

    # Open server graphs
    print("\nOpening analytical graphs...")
    for name, path in graphs.items():
        url = f"{API_BASE}{path}"
        webbrowser.open(url)
        print(f"  {name}: {url}")

if __name__ == "__main__":
    main()
PYEOF

chmod +x quantiq_client.py
./quantiq_client.py
