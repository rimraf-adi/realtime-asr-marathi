import sys
import time
import os
import subprocess
from pathlib import Path

train_log = Path("d:/realtime/experiments/e3_rl_guided_distill/logs/train.log")
step_target = 10000

print(f"[Monitor] Monitoring training up to step {step_target}...")
last_step = 0

while True:
    if train_log.exists():
        try:
            lines = train_log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            for line in reversed(lines[-25:]):
                if "Step " in line and "/" in line:
                    part = line.split("Step ")[1].split("/")[0]
                    curr_step = int(part)
                    if curr_step != last_step:
                        last_step = curr_step
                        print(f"[Monitor] {line}")
                    if curr_step >= step_target:
                        print(f"[Monitor] Target step {step_target} reached! Checking checkpoint...")
                        break
        except Exception as e:
            pass

    if last_step >= step_target:
        ckpt_path = Path(f"d:/realtime/experiments/e3_rl_guided_distill/checkpoints/causal_moe_step_{step_target}.pt")
        # Wait up to 60s for file write to complete
        for _ in range(30):
            if ckpt_path.exists() and ckpt_path.stat().st_size > 500 * 1024 * 1024:
                print(f"[Monitor] Checkpoint verified: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
                break
            time.sleep(2)
        
        print("[Monitor] Launching Full Evaluation Suite...")
        cmd = [
            "d:/realtime/.venv/Scripts/python.exe",
            "d:/realtime/tasks/run_e3_eval.py",
            "--checkpoint", str(ckpt_path),
            "--batch-size", "16",
        ]
        res = subprocess.run(cmd, cwd="d:/realtime")
        print(f"[Monitor] Eval suite finished with exit code {res.returncode}.")
        break

    time.sleep(15)
