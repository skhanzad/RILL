"""Package a Rill checkpoint for in-browser (phone/laptop/desktop) inference with ONNX Runtime Web (WASM, 1 thread):
ONNX export of the System-1 readout -> int8 embeddings + 8-bit weight-only MatMuls -> split into <=14 MB parts ->
tokenizer files + the ORT WASM binary copied next to the page (the page loads ORT's ES-module bundle, whose Emscripten
glue is built in, and hands it this binary). An RL calibrator, if the checkpoint has one, travels in the manifest and
is applied by rill-core.js.

    python scripts/export_demo.py --ckpt runs/df32-C/best.pt --out demo/site
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
    # The model travels as gzip, base64-encoded into text parts of <= part_mb MB (hosts that serve only web media types
    # accept text); every part decodes on its own because each holds a multiple of 3 gzip bytes.
    gz = gzip.compress(data, 9)
    n = int(a.part_mb * 1e6 * 3 / 4) // 3 * 3
    parts = []
    mdir = out / "model"; mdir.mkdir(exist_ok=True)
    for f in list(mdir.glob("rill.part*")):
        f.unlink()
    for i in range(0, len(gz), n):
        name = f"rill.part{len(parts)}.txt"
        (mdir / name).write_text(base64.b64encode(gz[i:i + n]).decode())
        parts.append({"file": f"model/{name}", "bytes": len(gz[i:i + n]), "chars": (mdir / name).stat().st_size})
    meta = json.loads((tmp / "rill_meta.json").read_text())
    manifest = {"parts": parts, "encoding": "gzip+base64", "gz_bytes": len(gz), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "meta": meta,
                "inputs": ["input_ids", "segment_ids", "position_ids", "slot_index"], "outputs": meta["outputs"],
                "checkpoint": str(a.ckpt)}
    (mdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tdir = out / "tokenizer"; tdir.mkdir(exist_ok=True)
    tok = json.loads((tmp / "tokenizer" / "tokenizer.json").read_text())
    (tdir / "tokenizer.json").write_text(json.dumps(tok, separators=(",", ":"), ensure_ascii=False))
    shutil.copy(tmp / "tokenizer" / "tokenizer_config.json", tdir / "tokenizer_config.json")
    odir = out / "ort"; odir.mkdir(exist_ok=True)
    for f in odir.glob("*.mjs"):
        f.unlink()
    shutil.copy(Path(a.ort_dist) / "ort-wasm-simd-threaded.wasm", odir / "ort-wasm-simd-threaded.wasm")
    manifest["runtime"] = {"file": "ort/ort-wasm-simd-threaded.wasm", "bytes": (odir / "ort-wasm-simd-threaded.wasm").stat().st_size,
                           "package": "onnxruntime-web@1.30.0"}
    (mdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"model_mb": len(data) / 1e6, "gzip_mb": len(gz) / 1e6, "parts": len(parts), "site_mb": total / 1e6}, indent=1))


if __name__ == "__main__":
    main()
