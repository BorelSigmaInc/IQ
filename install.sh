#!/bin/bash
set -e

INSTALL_DIR="$HOME/quantiq-client"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Fast path – skip pip if venv already exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -q requests matplotlib pandas numpy mplcursors
else
    source venv/bin/activate
fi

cat > quantiq_client.py << 'PYEOF'
#!/usr/bin/env python3
"""QuantIQ Client – interactive parallel coordinates + static graphs."""

import sys, os, json, webbrowser
import requests
import numpy as np
import matplotlib
# IMPORTANT: use interactive backend for the parallel coordinates window
matplotlib.use('TkAgg')   # works on Linux with python3-tk installed; fallback to Qt5Agg if needed
import matplotlib.pyplot as plt
import mplcursors
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
        print("Exiting. You can run the command again when you're ready.")
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

    # ---------- Interactive parallel coordinates ----------
    if plot_rows:
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

        # Hover behaviour
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
        plt.show()   # Opens interactive window; script pauses here until user closes it

        # After the window is closed, save a static copy
        save_path = SAVE_DIR / "parallel_coordinates.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"\nParallel coordinates graph saved to: {save_path}")

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

    # Offer to open the static graphs
    if saved_files or plot_rows:
        open_now = input("\nOpen the saved graphs now to review your leads? (Y/N): ").strip().upper()
        if open_now == "Y":
            # open the parallel coordinates static image
            pc_path = SAVE_DIR / "parallel_coordinates.png"
            if pc_path.exists():
                webbrowser.open(f"file://{pc_path}")
            for f in saved_files:
                webbrowser.open(f"file://{f}")

if __name__ == "__main__":
    main()
PYEOF

chmod +x quantiq_client.py
./quantiq_client.py </dev/tty
