"""Build an HTML viewer for SAE feature interpretation results."""
import json
import glob
import html
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
PATTERN = os.path.join(RESULTS_DIR, "sae_feature_interp_gemma-2-2b_20-gemmascope-res-16k_*.json")
OUTPUT = os.path.join(RESULTS_DIR, "sae_feature_viewer.html")

# Load all result files
files = sorted(glob.glob(PATTERN), key=lambda f: int(f.rsplit("_", 1)[1].replace(".json", "")))
features = []
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    features.append(data)

# Build HTML
def esc(s):
    return html.escape(str(s)) if s else ""

feature_tabs = []
feature_panels = []

for i, feat in enumerate(features):
    fidx = feat["feature_idx"]
    autointerp = feat.get("autointerp", "")
    # Truncate autointerp for tab label
    label_short = autointerp.split(":::")[0].strip()[:40] if autointerp else f"Feature {fidx}"
    url = feat.get("neuronpedia_url", "")

    feature_tabs.append(
        f'<button class="tab{"  active" if i == 0 else ""}" onclick="showFeature({i})" id="tab-{i}">'
        f'<span class="tab-idx">#{fidx}</span> {esc(label_short)}</button>'
    )

    rows = []
    for lr in feat.get("layer_results", []):
        layer = lr["inject_layer"]
        act = lr.get("best_activation", 0)
        contrastive = lr.get("decoded_contrastive", "")
        patchscopes = lr.get("decoded_patchscopes", "")

        # Color activation
        act_val = float(act) if act else 0
        if act_val > 100:
            act_class = "act-high"
        elif act_val > 30:
            act_class = "act-med"
        else:
            act_class = "act-low"

        rows.append(f"""<tr>
            <td class="layer-col">Layer {layer}</td>
            <td class="act-col {act_class}">{act_val:.1f}</td>
            <td class="text-col contrastive-col">{esc(contrastive)}</td>
            <td class="text-col patchscopes-col">{esc(patchscopes)}</td>
        </tr>""")

    panel_html = f"""<div class="panel" id="panel-{i}" style="display: {'block' if i == 0 else 'none'}">
        <div class="feature-header">
            <h2>Feature #{fidx}</h2>
            <a href="{esc(url)}" target="_blank" class="np-link">{esc(url)}</a>
            <p class="autointerp">{esc(autointerp)}</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th class="layer-col">Layer</th>
                    <th class="act-col">Activation</th>
                    <th class="text-col">Contrastive</th>
                    <th class="text-col">Patchscopes</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>"""
    feature_panels.append(panel_html)

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SAE Feature Interpretation Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }}
.container {{ display: flex; height: 100vh; }}
.sidebar {{
    width: 280px; min-width: 280px; background: #16213e; overflow-y: auto;
    border-right: 1px solid #333; padding: 8px 0;
}}
.sidebar h1 {{ font-size: 14px; padding: 8px 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
.tab {{
    display: block; width: 100%; text-align: left; padding: 8px 12px;
    background: none; border: none; color: #ccc; cursor: pointer;
    font-size: 13px; border-left: 3px solid transparent;
    transition: all 0.15s;
}}
.tab:hover {{ background: #1a1a3e; }}
.tab.active {{ background: #1a1a3e; border-left-color: #4fc3f7; color: #fff; }}
.tab-idx {{ color: #4fc3f7; font-weight: 600; margin-right: 4px; }}
.main {{ flex: 1; overflow-y: auto; padding: 24px; }}
.feature-header {{ margin-bottom: 16px; }}
.feature-header h2 {{ font-size: 22px; margin-bottom: 4px; }}
.np-link {{ color: #4fc3f7; font-size: 13px; text-decoration: none; }}
.np-link:hover {{ text-decoration: underline; }}
.autointerp {{
    margin-top: 8px; padding: 10px 14px; background: #16213e;
    border-radius: 6px; font-size: 14px; line-height: 1.5; color: #aaa;
    border-left: 3px solid #4fc3f7;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
thead th {{
    position: sticky; top: 0; background: #16213e; padding: 10px 12px;
    text-align: left; font-weight: 600; border-bottom: 2px solid #333;
    color: #aaa; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px;
}}
td {{ padding: 10px 12px; border-bottom: 1px solid #2a2a4a; vertical-align: top; }}
.layer-col {{ width: 80px; font-weight: 600; white-space: nowrap; }}
.act-col {{ width: 90px; text-align: right; font-family: monospace; font-weight: 600; }}
.text-col {{ line-height: 1.5; }}
.act-high {{ color: #ef5350; }}
.act-med {{ color: #ffa726; }}
.act-low {{ color: #66bb6a; }}
tr:hover {{ background: rgba(79, 195, 247, 0.05); }}
.contrastive-col {{ border-right: 1px solid #2a2a4a; }}

/* Keyboard nav hint */
.hint {{ font-size: 11px; color: #555; padding: 8px 12px; border-top: 1px solid #333; }}
</style>
</head>
<body>
<div class="container">
    <div class="sidebar">
        <h1>Features</h1>
        {"".join(feature_tabs)}
        <div class="hint">Use arrow keys (up/down) to navigate</div>
    </div>
    <div class="main">
        {"".join(feature_panels)}
    </div>
</div>
<script>
let current = 0;
const n = {len(features)};
function showFeature(i) {{
    document.getElementById('panel-' + current).style.display = 'none';
    document.getElementById('tab-' + current).classList.remove('active');
    current = i;
    document.getElementById('panel-' + current).style.display = 'block';
    document.getElementById('tab-' + current).classList.add('active');
    document.getElementById('tab-' + current).scrollIntoView({{ block: 'nearest' }});
}}
document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowDown' || e.key === 'j') {{ e.preventDefault(); showFeature((current + 1) % n); }}
    if (e.key === 'ArrowUp' || e.key === 'k') {{ e.preventDefault(); showFeature((current - 1 + n) % n); }}
}});
</script>
</body>
</html>"""

with open(OUTPUT, "w") as f:
    f.write(html_content)

print(f"Wrote {OUTPUT} with {len(features)} features")
