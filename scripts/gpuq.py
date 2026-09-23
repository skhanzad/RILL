"""A tiny sequential GPU job queue. Jobs are lines in logs/gpu_queue.txt ("name | shell command"); finished job names
are appended to logs/gpu_done.txt. New lines can be appended while the queue runs. A line "STOP" ends the runner."""
import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# optional: python scripts/gpuq.py <queue file> <done file>  (a second runner for small jobs that fit next to the first)
QUEUE = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "logs/gpu_queue.txt")
DONE = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "logs/gpu_done.txt")
LOGS = ROOT / "logs/jobs"
LOGS.mkdir(parents=True, exist_ok=True)
QUEUE.touch(); DONE.touch()


def pending():
    done = {l.split("\t")[0] for l in DONE.read_text().splitlines() if l.strip()}
    for line in QUEUE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "STOP":
            return "STOP"
        name, _, cmd = line.partition("|")
        name, cmd = name.strip(), cmd.strip()
        if name not in done:
            return name, cmd
    return None


while True:
    job = pending()
    if job == "STOP":
        break
    if job is None:
        time.sleep(20); continue
    name, cmd = job
    start = time.time()
    print(f"[{datetime.datetime.now():%H:%M:%S}] start {name}: {cmd}", flush=True)
    with (LOGS / f"{name}.log").open("w") as f:
        rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    with DONE.open("a") as f:
        f.write(f"{name}\t{rc}\t{time.time()-start:.0f}s\t{datetime.datetime.now():%H:%M:%S}\n")
    print(f"[{datetime.datetime.now():%H:%M:%S}] done {name} rc={rc} {time.time()-start:.0f}s", flush=True)
