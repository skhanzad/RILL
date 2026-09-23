#!/bin/bash
# CPU latency (4 threads, pinned to cores 24-31), identical requests for every system.
PY=third_party/kev/.venv/bin/python
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=scripts/nofla
B="taskset -c 24-31 $PY scripts/bench_latency.py --device cpu --threads 4 --n 20"
$B --system semif --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --dtype float32 --out results/latency/cpu4-semif-qwen3-0.6b.json
$B --system kev --model jaredpalmer/kev-0.8b --dtype float32 --out results/latency/cpu4-kev-0.8b.json
$B --system semif --model openbmb/MiniCPM5-2B --revision 12a3808a956f869c767195e9266b59c4d21d92e2 --dtype float32 --n 10 --out results/latency/cpu4-semif-minicpm5-2b.json
$B --system semif --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --dtype float32 --n 10 --out results/latency/cpu4-semif-qwen3.5-4b.json
$B --system df --ckpt runs/df68-A/best.pt --out results/latency/cpu4-rill-68m-A.json
$B --system df --ckpt runs/df68-A/best.pt --int8 --out results/latency/cpu4-rill-68m-A-int8.json
