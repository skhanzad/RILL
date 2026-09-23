#!/bin/bash
# CPU latency for the final Rill models (served configuration: Stage 1 + RL calibrator) -- same protocol as cpu_bench.sh:
# PyTorch fp32, and the deployable ONNX export with int8 embeddings and 8-bit weight-only matrices (ONNX Runtime).
#   bash scripts/cpu_bench2.sh 32 68 150
PY=third_party/kev/.venv/bin/python
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=scripts/nofla
B="taskset -c 24-31 $PY scripts/bench_latency.py --device cpu --threads 4 --n 20"
for m in "${@:-32 68 150}"; do
  $B --system df --ckpt runs/rlcal${m}-C/best.pt --out results/latency/cpu4-rill-${m}m-C.json
  $B --system onnx --onnx export/rill-${m}m-C-rl/site_build/rill.g8w8.onnx --export_dir export/rill-${m}m-C-rl/site_build \
     --out results/latency/cpu4-rill-${m}m-C-g8w8.json
done
