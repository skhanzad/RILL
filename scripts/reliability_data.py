"""Reliability-diagram data (10 equal-width confidence bins, top-label) on transfer-v4 test for selected systems."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import SUITES  # noqa: F401
from kev.metrics import scored_rows, tempered_row
reg = json.loads((ROOT / "results/registry.json").read_text())
out = ROOT / "paper/figures/data"; out.mkdir(parents=True, exist_ok=True)
for name, tag in [("Rill-68M", "rill"), ("Rill-68M+RL", "rillrl"), ("Kev-0.8B", "kev"), ("Kev-4B", "kevfour"), ("SemIf-Qwen3.5-4B", "semif")]:
    p = ROOT / reg[name]["dir"] / "transfer-v4.test.rows.json"
    if not p.exists():
        continue
    rows = scored_rows(json.loads(p.read_text()))
    if reg[name].get("shipped_T"):          # Kev as released: its checkpoints ship a temperature
        rows = [tempered_row(r, reg[name]["shipped_T"]) if "logits" in r else r for r in rows]
    conf = np.array([max(r["p"]) for r in rows]); ok = np.array([int(np.argmax(r["p"]) == r["label"]) for r in rows])
    lines = ["conf acc n"]
    for lo in np.linspace(0, 0.9, 10):
        m = (conf >= lo) & ((conf < lo + 0.1) if lo < 0.89 else (conf <= 1.0))
        if m.sum() >= 5:
            lines.append(f"{conf[m].mean():.4f} {ok[m].mean():.4f} {int(m.sum())}")
    (out / f"rel_{tag}.dat").write_text("\n".join(lines) + "\n")
    print(name, len(lines) - 1, "bins")
