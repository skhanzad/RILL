"""Pick the RL epoch with the best summed development objective (-mean task NLL on decision-v7 dev + transfer-v4 dev,
as logged by decisionflow.rl) and copy it to best.pt. Development partitions only; locked tests are never read."""
import json, shutil, sys
from pathlib import Path
run = Path(sys.argv[1])
hist = [json.loads(l) for l in (run / "rl.log").read_text().splitlines() if l.startswith("{")]
cands = [h for h in hist if h["tag"].startswith("epoch")]
best = max(cands, key=lambda h: h["decision-v7"]["objective"] + h["transfer-v4"]["objective"])
shutil.copy(run / f"{best['tag']}.pt", run / "best.pt")
(run / "selected.json").write_text(json.dumps({"selected": best["tag"], "objective": best["decision-v7"]["objective"] + best["transfer-v4"]["objective"]}, indent=1))
print(run.name, "->", best["tag"])
