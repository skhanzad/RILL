"""Dump reference packed layouts (Python) for the JS parity check."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from decisionflow.data import SUITES
from decisionflow.evaluate import request_to_rec
from decisionflow.layout import Layout
from kev.data import api_request
from kev.suite import load_split
tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
lay = Layout(tok, max_state=1500)
out = []
for suite, split in (("decision-v7", "development"), ("transfer-v4", "development"), ("scienthoon-v1", "development"), ("semif-v1", "development")):
    for r in load_split(SUITES[suite], split)[:150]:
        rec, meta = request_to_rec(r)
        e = lay.encode(rec)
        out.append({"request": api_request(r), "ids": e.ids.tolist(), "seg": e.seg.tolist(), "pos": e.pos.tolist(), "slots": e.slots})
meta = {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "q_id": lay.q_id, "opt_id": lay.opt_id, "max_option_tokens": 64, "max_instr_tokens": 256}
Path(sys.argv[1]).write_text(json.dumps({"meta": meta, "cases": out}))
print(len(out), "cases")
