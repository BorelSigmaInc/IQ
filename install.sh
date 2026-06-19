#!/bin/bash
set -e

INSTALL_DIR="$HOME/quantiq-client"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Always ensure the virtual environment and all dependencies are ready
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q requests matplotlib pandas numpy plotly

cat > quantiq_client.py << 'PYEOF'
#!/usr/bin/env python3
"""QuantIQ Client – interactive parallel coordinates (Plotly) + static graphs."""

import sys, os, json, webbrowser
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')   # used only for static PNGs
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from pathlib import Path

CONFIG_FILE = Path.home() / ".quantiq_config.json"
API_BASE = "http://46.224.200.113:8001"
SAVE_DIR = Path.home() / "quantiq-client" / "graphs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
    print("=" * 60)
    print("  Q U A N T I Q   L E A D   I N T E L L I G E N C E")
    print("=" * 60)

    # 1. Consent
    consent = input("Proceed to score leads using QuantIQ API? (Y/N): ").strip().upper()
    if consent != "Y":
        print("Exiting.")
        return
    print()

    # 2. API key
    key = input("Enter your QuantIQ API key: ").strip()
    save_key(key)
    print("Key saved.\n")

    # 3. Optional data file
    filename = input("Optional: Data file name (e.g., my_leads.csv) or press Enter to skip: ").strip()
    if filename:
        filetype = input("Type of file (csv/json/tsv/xlsx): ").strip().lower()
        folder = input("Path of folder containing the file: ").strip()
        full_path = os.path.join(folder, filename)
        if validate_file(full_path):
            file_consent = input(f"Allow QuantIQ to access '{full_path}'? (Y/N): ").strip().upper()
            if file_consent == "Y":
                print("File accepted. (Upload feature coming soon – using demo leads for now.)")
            else:
                print("Consent denied. Using pre-loaded leads.")
        else:
            print("File invalid. Continuing with demo leads.")
    else:
        print("No file provided. Using pre-loaded leads from QuantIQ server.")

    # 4. Number of leads
    limit = 10
    ans = input("\nHow many leads would you like to score? (default 10): ").strip()
    if ans.isdigit():
        limit = int(ans)

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

    # ---------- Interactive parallel coordinates (Plotly HTML) ----------
    if plot_rows:
        df = pd.DataFrame(plot_rows, columns=[
            'Priority', 'LeadID', 'Intent', 'Kernel', 'RBF', 'Gap', 'Unc', 'Ent', 'QFeat'
        ])
        fig = px.parallel_coordinates(
            df,
            dimensions=['Priority', 'Kernel', 'RBF', 'Gap', 'Unc', 'Ent', 'QFeat'],
            color='Priority',
            color_continuous_scale=px.colors.sequential.Blues,
            labels={col: col for col in df.columns},
            title='Interactive Parallel Coordinates – Hover to see Lead ID'
        )
        fig.update_traces(
            hovertemplate='<br>'.join([
                'LeadID: %{customdata[0]}',
                'Intent: %{customdata[1]}',
            ]),
            customdata=df[['LeadID', 'Intent']].values
        )
        html_path = SAVE_DIR / "parallel_coordinates.html"
        fig.write_html(str(html_path))
        print(f"\nInteractive parallel coordinates saved to: {html_path}")

        # Static PNG version
        data_arr = np.array(plot_rows)
        axes_idx = [0, 3, 4, 5, 6, 7, 8]
        axes_names = ['Priority','Kernel','RBF','Gap','Unc','Ent','QFeat']
        n_axes = len(axes_idx)
        n_leads = data_arr.shape[0]
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

        fig_static, ax = plt.subplots(figsize=(12,6))
        for i in range(n_leads):
            cmap = plt.cm.Blues if intent_vals[i] == 1 else plt.cm.Oranges
            c_intensity = 0.2 + 0.8 * (priority_vals[i] - priority_vals.min()) / (priority_vals.max() - priority_vals.min())
            ax.plot(range(n_axes), X[i], marker='o', markersize=2, linewidth=0.8, alpha=0.7, color=cmap(c_intensity))
        ax.set_xticks(range(n_axes))
        ax.set_xticklabels(axes_names)
        ax.set_title('Parallel Coordinates – Lead Profiles')
        static_path = SAVE_DIR / "parallel_coordinates.png"
        plt.tight_layout()
        plt.savefig(static_path, dpi=120)
        plt.close(fig_static)
        print(f"Static version saved to: {static_path}")

    # ---------- Download server graphs ----------
    saved_files = []
    print("\nServer analytical graphs (2D, 3D, comparison):")
    for name, rel_path in graphs.items():
        url = f"{API_BASE}{rel_path}"
        try:
            r = requests.get(url)
            if r.status_code == 200:
                fname = f"{name}_{usage.get('call_number','0')}.png"
                filepath = SAVE_DIR / fname
                with open(filepath, "wb") as f:
                    f.write(r.content)
                saved_files.append(str(filepath))
                print(f"  {name}: saved to {filepath}")
            else:
                print(f"  {name}: could not download (status {r.status_code})")
        except Exception as e:
            print(f"  {name}: error - {e}")

    print("\nAll graphs saved in:", SAVE_DIR)

    # Offer to open the interactive HTML and static graphs
    open_now = input("\nOpen the graphs now to review your leads? (Y/N): ").strip().upper()
    if open_now == "Y":
        html_path = SAVE_DIR / "parallel_coordinates.html"
        if html_path.exists():
            webbrowser.open(f"file://{html_path}")
        for f in saved_files:
            webbrowser.open(f"file://{f}")

if __name__ == "__main__":
    main()
PYEOF

chmod +x quantiq_client.py
./quantiq_client.py </dev/tty
