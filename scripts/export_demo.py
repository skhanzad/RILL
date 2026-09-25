"""Package a Rill checkpoint for in-browser (phone/laptop/desktop) inference with ONNX Runtime Web (WASM, 1 thread):
ONNX export of the System-1 readout -> int8 embeddings + 8-bit weight-only MatMuls -> gzip -> base64 parts of
<= part_mb MB. Every file the page needs (manifest, model parts, tokenizer, the gzipped ORT WASM binary) is written as a
JavaScript data script  RillData.put(key, JSON);  that the page loads with <script src>: a sandboxed viewer runs the
page at an opaque origin, where fetch() of the page's own files would need CORS headers. ORT's ES-module bundle (its
Emscripten glue is built in) is copied next to the page as ort/ort.wasm.bundle.min.js; the page imports it, or the same
file from a CDN at an opaque origin, and hands it the WASM binary. An RL calibrator, if the checkpoint has one, travels in
the manifest and is applied by rill-core.js. The output directory is what GitHub Pages serves (.github/workflows/pages.yml).

    python scripts/export_demo.py --ckpt runs/rlcal32-C/best.pt --out demo/site --ort_dist <onnxruntime-web>/package/dist
"""
import argparse
import base64
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ort_dist", required=True, help="onnxruntime-web 1.30.0 package/dist directory")
    ap.add_argument("--part_mb", type=float, default=14.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.name + "_build"); tmp.mkdir(exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "decisionflow.export", "--ckpt", a.ckpt, "--out", str(tmp)], cwd=ROOT)
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    g8 = tmp / "gather8.onnx"
    quantize_dynamic(str(tmp / "rill_readout.onnx"), str(g8), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    m = onnx.load(str(g8))
    q = MatMulNBitsQuantizer(m, algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8))
    q.process()
    final = tmp / "rill.g8w8.onnx"
    q.model.save_model_to_file(str(final), use_external_data_format=False)
    data = final.read_bytes()
    sys.path.insert(0, str(ROOT))
    from decisionflow.export import write_data_script
    # old-format files (fetched JSON / text / wasm) are replaced by data scripts
    for pat in ("model/rill.part*", "model/part*", "model/manifest.*", "model/rill-js.json", "tokenizer/*", "ort/*"):
        for f in out.glob(pat):
            f.unlink()
    mdir, tdir, odir = out / "model", out / "tokenizer", out / "ort"
    for d in (mdir, tdir, odir):
        d.mkdir(exist_ok=True)
    # model: gzip, base64 in parts; each part holds a multiple of 3 gzip bytes, so it decodes on its own
    gz = gzip.compress(data, 9)
    n = int(a.part_mb * 1e6 * 3 / 4) // 3 * 3
    parts = []
    for i in range(0, len(gz), n):
        key = f"part{len(parts)}"
        size = write_data_script(mdir / f"{key}.js", key, base64.b64encode(gz[i:i + n]).decode())
        parts.append({"file": f"model/{key}.js", "key": key, "bytes": len(gz[i:i + n]), "chars": size})
    # tokenizer (the page needs only tokenizer.json)
    tok = json.loads((tmp / "tokenizer" / "tokenizer.json").read_text())
    tok_size = write_data_script(tdir / "tokenizer.js", "tokenizer", tok)
    # ONNX Runtime Web's WASM binary, gzipped (14 MB -> about 5 MB as base64)
    wasm = (Path(a.ort_dist) / "ort-wasm-simd-threaded.wasm").read_bytes()
    wasm_size = write_data_script(odir / "wasm.js", "wasm", base64.b64encode(gzip.compress(wasm, 9)).decode())
    shutil.copyfile(Path(a.ort_dist) / "ort.wasm.bundle.min.mjs", odir / "ort.wasm.bundle.min.js")   # the glue for that binary
    ort_pkg = json.loads((Path(a.ort_dist).parent / "package.json").read_text())
    meta = json.loads((tmp / "rill_meta.json").read_text())
    manifest = {"parts": parts, "encoding": "gzip+base64", "gz_bytes": len(gz), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "meta": meta, "inputs": ["input_ids", "segment_ids", "position_ids", "slot_index"], "outputs": meta["outputs"],
                "checkpoint": str(a.ckpt),
                "tokenizer": {"file": "tokenizer/tokenizer.js", "key": "tokenizer", "chars": tok_size},
                "runtime": {"file": "ort/wasm.js", "key": "wasm", "bytes": len(wasm), "chars": wasm_size, "encoding": "gzip+base64",
                            "package": f"{ort_pkg['name']}@{ort_pkg['version']}"},
                "engine": {"file": "model/engine.js", "key": "engine"}}
    write_data_script(mdir / "manifest.js", "manifest", manifest)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"model_mb": len(data) / 1e6, "gzip_mb": len(gz) / 1e6, "parts": len(parts), "site_mb": total / 1e6}, indent=1))


if __name__ == "__main__":
    main()
