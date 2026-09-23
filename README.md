# Rill: tiny flow-matching decision models

Rill answers **typed decisions** (yes/no, multiple choice, ordered scores) about a text or JSON *state*. It returns a
calibrated probability distribution per question and never generates text. It uses the System One request/response
shape that open decision-model projects share (state + named questions with `noul` / `choice` / `score` types).

* **Model** (`decisionflow/model.py`): a pretrained Ettin/ModernBERT encoder, 32M/68M/150M parameters. Its lower
  layers plus segment-masked ConvNeXt V2 blocks form a text branch that runs once. Its top layers become DiT blocks,
  modulated by adaLN on the flow time, over an *answer track* that lives on the `[OPT]` option slots. The training
  objective is conditional flow matching on one-hot answers (CDF answers for scores). A single pass at t = 0 returns
  calibrated probabilities (the "System 1" readout). An Euler ODE with self-conditioning is the "System 2" sampler.
* **RL post-training** (`decisionflow/rl.py`): the stochastic flow sampler draws decisions, only the outcome
  `z = 1[decision correct]` is observed, and the reward is `R = z - (c - z)^2` for the stated confidence `c`. The
  default (`--calibrator 1 --lam_pg 0 --beta 0`) freezes the network and trains a 25-50k-parameter, arg-max-preserving
  temperature head `tau(x)` on a pool of novel public tasks through the reward's confidence term (the GRPO term is
  available with `--lam_pg 1`; development NLL preferred 0). Full-network variants: `--calibrator 0`.
* **Evaluation**: the frozen public suites of Kev (`third_party/kev/evals`), scored with Kev's unmodified metric code,
  against Kev, SemIf (and JEV-CPU's configuration), the Layla on-device models, and NLI zero-shot classifiers.
  Nothing here calls a hosted decision API.

## Layout

```
decisionflow/        model, token layout, data pipeline, Stage-1 training, RL, evaluation, ONNX export
baselines/           predictors and runner for SemIf / Layla (letter logits + generative) / NLI / Kev
scripts/             data building, evaluation, aggregation, tables, latency, quantisation checks, demo packaging
demo/site/           the in-browser demo (static files); demo/*.mjs are its Node checks and benchmark
paper/               AISTATS 2027 paper (main.tex, TikZ architecture figure, sections/, refs.bib)
results/             prediction rows, reports, latency, aggregate.json (every number in the paper)
runs/                checkpoints and training logs
third_party/         kev (Apache-2.0) and SemIf (MIT) clones used for data, prompts and metrics
```

## Reproduce

Everything runs in the environment pinned by Kev (`third_party/kev`: Python 3.13, torch 2.8, transformers 5.17):

```bash
cd third_party/kev && uv sync --extra serve && cd ../..
PY=third_party/kev/.venv/bin/python

# Training data: A = Kev's exact data; B = the same sources at 26x scale; C = B + public NLI/paraphrase/MC data
$PY scripts/build_expB.py --out data/expB --structures 200
$PY scripts/build_expC.py --out data/expC
$PY scripts/build_rlpool.py --out data/rlpool            # the eight novel tasks for post-training

# Stage 1 (flow matching) and the RL calibrator, e.g. the 68M model on C
$PY -m decisionflow.train --backbone jhu-clsp/ettin-encoder-68m --train data/expC/train.jsonl --epochs 1 --max_tokens 12288 --out runs/df68-C
$PY -m decisionflow.rl --init runs/df68-C/best.pt --pool data/rlpool/train.jsonl --epochs 3 --max_tokens 1500 --G 6 \
    --calibrator 1 --beta 0 --lam_pg 0 --out runs/rlcal68-C
$PY scripts/select_epoch.py runs/rlcal68-C              # development partitions only

# Evaluation (add --canonical 1 to sort choice options by name: exact order invariance), baselines, tables
$PY scripts/eval_rill.py --ckpt runs/rlcal68-C/best.pt --out results/rill/rill-68m-C-rl
$PY baselines/run.py --system semif --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --out results/baselines/semif-qwen3.5-4b
```

`logs/gpu_queue.txt` lists every job that produced the reported results, in order (`logs/gpu_queue2.txt` ran the
small baselines in parallel). Kev-4B runs with `KEV_MERGE=0 KEV_DTYPE=bf16`: merging its adapter in fp32 needs more than
16 GB.

## On-device: CPU, ONNX and the browser

```bash
# ONNX export of the System-1 readout (+ the calibrator weights), int8 embeddings + 8-bit weight-only MatMulNBits,
# packaged for the browser as JavaScript data scripts (model parts, tokenizer, gzipped ONNX Runtime WASM binary)
$PY scripts/export_demo.py --ckpt runs/rlcal32-C/best.pt --out demo/site --ort_dist <onnxruntime-web-1.30.0>/package/dist
$PY scripts/export_js_engine.py --ckpt runs/rlcal32-C/best.pt --onnx demo/site_build/rill.g8w8.onnx --out demo/site/model
$PY scripts/check_quant.py --ckpt runs/rlcal32-C/best.pt --onnx_dir demo/site_build --variants int8 --extra g8w8=demo/site_build/rill.g8w8.onnx
bash scripts/cpu_bench2.sh 32 68 150                     # CPU latency, 4 threads: PyTorch fp32 and the 8-bit ONNX model
node demo/engine_check.mjs <ort.wasm.bundle.min.mjs> <ort-wasm-simd-threaded.wasm> <ref.json>   # JS engine vs ORT vs PyTorch
node demo/bench_wasm.mjs <ort.wasm.bundle.min.mjs> <ort-wasm-simd-threaded.wasm> <lat_ref.json> results/latency/wasm-rill-32m-C.json
cd demo/site && python3 -m http.server 8000             # then open http://localhost:8000
```

The page (`demo/site/index.html`) runs Rill-32M with its calibrator on one WebAssembly thread through ONNX Runtime
Web. Where WebAssembly is not allowed (or with `#js` in the address) it switches to `rill-engine.js`, a dependency-free
JavaScript implementation of the same forward pass that reads the same 8-bit weights out of the ONNX file, in a Web
Worker. `rill-core.js` holds the tokenizer, request packing and the calibrator, and matches the Python layout
token for token. Weight-only 8-bit quantisation keeps about 99% of the fp32 decisions (`demo/site_build/quant_check.*`).
Every data file is a script of the form `RillData.put(key, JSON)` loaded with `<script src>`, not `fetch()`: viewers
that sandbox the page give it an opaque origin, where fetching its own files would need CORS headers.
