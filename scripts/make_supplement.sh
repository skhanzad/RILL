#!/bin/bash
# Anonymous supplementary bundle: code, paper sources, result summaries and the job log (no checkpoints, raw
# prediction rows, training data or third-party clones; kev and SemIf are referenced by URL in the README).
set -e
OUT=${1:-supplement.zip}
rm -f "$OUT"
zip -qr "$OUT" README.md decisionflow baselines scripts demo/*.mjs demo/*.py \
    demo/site/index.html demo/site/rill-core.js demo/site/rill-engine.js demo/site/engine-worker.js \
    paper/main.tex paper/numbers.tex paper/numbers_auto.tex paper/refs.bib paper/refs_verification.md paper/aistats2027.sty \
    paper/sections paper/tables paper/figures/arch.tex paper/figures/arch_defs.tex paper/figures/pareto.tex \
    paper/figures/reliability.tex paper/figures/data \
    results/registry.json results/aggregate.json results/ablations.json results/latency results/pooltest/report-68B.json \
    logs/gpu_queue.txt logs/gpu_queue2.txt \
    -x "*/__pycache__/*" "*.pyc" "demo/ort.wasm.bundle.min.mjs" "demo/ort-wasm-simd-threaded.wasm"
unzip -l "$OUT" | tail -1
