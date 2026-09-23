#!/bin/bash
PY=third_party/kev/.venv/bin/python
T=baselines/templates
run() { name=$1; shift; echo "=== $name"; $PY baselines/run.py "$@" --suites transfer-v4,scienthoon-v1 --limit 3 --out results/smoke/$name 2>&1 | grep -E "^\{|FAIL|Error|error|Traceback" | head -8; }
run semif-q06 --system semif --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
run semif-q35 --system semif --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
run semif-minicpm --system semif --model openbmb/MiniCPM5-2B --revision 12a3808a956f869c767195e9266b59c4d21d92e2
run kev08 --system kev --model jaredpalmer/kev-0.8b
run layla-tiny --system letters --model l3utterfly/tinyllama-1.1b-layla-v4 --chat_template $T/chatml.jinja
run layla-qwen --system letters --model l3utterfly/Qwen1.5-1.8B-layla-v4
run layla-phi2 --system letters --model l3utterfly/phi-2-layla-v1 --chat_template $T/user_assistant.jinja --letter_prefix " "
run layla-mistral --system letters --model l3utterfly/mistral-7b-v0.1-layla-v4 --quant nf4 --chat_template $T/user_assistant.jinja --letter_prefix " "
run gen-phi2 --system generative --model l3utterfly/phi-2-layla-v1 --chat_template $T/user_assistant.jinja --letter_prefix " "
run nli-base --system nli --model MoritzLaurer/deberta-v3-base-zeroshot-v2.0
run nli-bart --system nli --model facebook/bart-large-mnli
