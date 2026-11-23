"""
Master Sequential 3-Stage Pipeline Runner + Evaluation for Causal Streaming Conformer (130M MoE).
Aligned with ../marathi-asr/scripts/pipeline.py & run_ablation_queue.py:
  Stage 1: stage1_causal_pretrain (30,000 steps)   -> experiments/stage1_causal_pretrain/checkpoints/best_model.pt
  Stage 2: stage2_causal_moe      (80,000 steps)   -> experiments/stage2_causal_moe/checkpoints/best_model.pt
  Stage 3: stage3_final_alignment (40,000 steps)   -> experiments/stage3_final_alignment/checkpoints/best_model.pt
  Stage 4: e3_full_eval_suite     (RESPIN, KathBath, FLEURS, 160ms streaming RTF)

Maintains d:/realtime/experiments/pipeline_status.json, supports automatic checkpoint resume
from the newest readable step checkpoint, and keeps only the newest 3 rolling step checkpoints.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
STATUS_FILE = EXPERIMENTS_DIR / "pipeline_status.json"
MARATHI_ENV_FILE = Path("d:/marathi-asr/.env")
PYTHON_EXE = sys.executable


def load_env_with_hf() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    if MARATHI_ENV_FILE.exists():
        for raw in MARATHI_ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    return env


def update_status(stage_name: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if STATUS_FILE.exists():
        try:
            data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[stage_name] = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        **(details or {}),
    }
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_stage_completed(stage_name: str) -> bool:
    if not STATUS_FILE.exists():
        return False
    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return data.get(stage_name, {}).get("status") == "completed"
    except Exception:
        return False


def step_files(ckpt_dir: Path, pattern: str) -> List[Path]:
    if not ckpt_dir.exists():
        return []
    return sorted(
        ckpt_dir.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else 0,
    )


def prune_checkpoints(ckpt_dir: Path, pattern: str, keep_last: int = 3) -> None:
    for old_f in step_files(ckpt_dir, pattern)[:-keep_last]:
        try:
            old_f.unlink()
        except OSError:
            pass


def latest_step_ckpt(ckpt_dir: Path, pattern: str) -> Optional[Path]:
    """Return newest readable step checkpoint in ckpt_dir, skipping partial/corrupt files."""
    import torch

    for f in reversed(step_files(ckpt_dir, pattern)):
        try:
            torch.load(f, map_location="cpu", weights_only=False)
            return f
        except Exception as exc:
            print(f"[Pipeline] SKIP unreadable checkpoint {f.name}: {type(exc).__name__}", flush=True)
    return None


def resolve_stage_checkpoint(ckpt_dir: Path, pattern: str) -> Optional[Path]:
    best = ckpt_dir / "best_model.pt"
    if best.exists():
        return best
    return latest_step_ckpt(ckpt_dir, pattern)


def run_stage(
    stage_name: str,
    cmd: List[str],
    log_path: Path,
    completion_marker: str,
    expected_ckpt: Optional[Path] = None,
    prune_dir: Optional[Path] = None,
    prune_pattern: Optional[str] = None,
) -> None:
    print(f"\n{'=' * 70}\n[Pipeline] Launching {stage_name}: {' '.join(cmd)}\n{'=' * 70}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    update_status(stage_name, "running", {"cmd": cmd, "log_file": str(log_path)})

    env = load_env_with_hf()
    t0 = time.time()

    with open(log_path, "a", encoding="utf-8", errors="replace") as f_out:
        f_out.write(f"\n--- Pipeline launching {stage_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f_out.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=f_out,
            stderr=subprocess.STDOUT,
            env=env,
        )
        while True:
            try:
                proc.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                if prune_dir and prune_pattern:
                    prune_checkpoints(prune_dir, prune_pattern, keep_last=3)

    if prune_dir and prune_pattern:
        prune_checkpoints(prune_dir, prune_pattern, keep_last=3)

    elapsed_min = round((time.time() - t0) / 60.0, 1)
    ret = proc.returncode
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    ckpt_ok = expected_ckpt is None or expected_ckpt.exists()

    if ret != 0:
        if ckpt_ok and completion_marker in log_text:
            print(
                f"[Pipeline] {stage_name} logged completion despite teardown exit code {ret}; treating as completed.",
                flush=True,
            )
        else:
            update_status(stage_name, "failed", {"exit_code": ret, "elapsed_min": elapsed_min})
            raise RuntimeError(f"Stage '{stage_name}' failed (exit code {ret}) after {elapsed_min} min. Check {log_path}")

    update_status(stage_name, "completed", {"exit_code": 0, "elapsed_min": elapsed_min})
    print(f"[Pipeline] {stage_name} completed in {elapsed_min} minutes.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Master 3-Stage + Eval Full-Scale Causal MoE Pipeline")
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--to-stage", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--stage1-steps", type=int, default=300000)
    parser.add_argument("--stage2-steps", type=int, default=300000)
    parser.add_argument("--stage3-steps", type=int, default=300000)
    parser.add_argument("--left-context", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Pipeline] Full-scale Causal MoE Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # ------------------------------------------------------------------
    # Stage 1: Causal Acoustic Self-Supervised Pretraining (up to 300k steps + Early Stopping)
    # ------------------------------------------------------------------
    s1_dir = EXPERIMENTS_DIR / "stage1_causal_pretrain"
    s1_ckpt_dir = s1_dir / "checkpoints"
    if args.from_stage <= 1 <= args.to_stage:
        if is_stage_completed("stage1_causal_pretrain") and (s1_ckpt_dir / "best_model.pt").exists():
            print("[Pipeline] Stage 1 already completed; skipping.", flush=True)
        else:
            s1_resume = latest_step_ckpt(s1_ckpt_dir, "causal_pretrain_step_*.pt")
            cmd = [
                PYTHON_EXE, "-u", "tasks/run_stage1_pretrain.py",
                "--steps", str(args.stage1_steps),
                "--batch-size", "32",
                "--lr", "4e-4",
                "--warmup-steps", "1000",
                "--save-every", "1000",
                "--log-every", "25",
                "--keep-last-k", "3",
                "--left-context", str(args.left_context),
                "--early-stop-patience", str(args.early_stop_patience),
            ]
            if s1_resume:
                cmd.extend(["--resume", str(s1_resume)])
            run_stage(
                stage_name="stage1_causal_pretrain",
                cmd=cmd,
                log_path=s1_dir / "logs" / "pipeline_stage1.log",
                completion_marker="Stage 1 pretraining completed",
                expected_ckpt=s1_ckpt_dir / "best_model.pt",
                prune_dir=s1_ckpt_dir,
                prune_pattern="causal_pretrain_step_*.pt",
            )

    # ------------------------------------------------------------------
    # Stage 2: Causal MoE Dialect Adaptation (up to 300k steps + Early Stopping)
    # ------------------------------------------------------------------
    s2_dir = EXPERIMENTS_DIR / "stage2_causal_moe"
    s2_ckpt_dir = s2_dir / "checkpoints"
    if args.from_stage <= 2 <= args.to_stage:
        if is_stage_completed("stage2_causal_moe") and (s2_ckpt_dir / "best_model.pt").exists():
            print("[Pipeline] Stage 2 already completed; skipping.", flush=True)
        else:
            s1_ckpt = resolve_stage_checkpoint(s1_ckpt_dir, "causal_pretrain_step_*.pt")
            if not s1_ckpt:
                raise FileNotFoundError(f"Stage 1 checkpoint missing in {s1_ckpt_dir}")
            s2_resume = latest_step_ckpt(s2_ckpt_dir, "causal_moe_stage2_step_*.pt")
            cmd = [
                PYTHON_EXE, "-u", "tasks/run_stage2_moe_adapt.py",
                "--stage1-ckpt", str(s1_ckpt),
                "--steps", str(args.stage2_steps),
                "--batch-size", "32",
                "--grad-accum-steps", "4",
                "--lr", "1.5e-4",
                "--warmup-steps", "1000",
                "--ctc-warmup-steps", "3000",
                "--phase4a-steps", "15000",
                "--aux-weight", "0.01",
                "--save-every", "1000",
                "--log-every", "25",
                "--keep-last-k", "3",
                "--left-context", str(args.left_context),
                "--early-stop-patience", str(args.early_stop_patience),
            ]
            if s2_resume:
                cmd.extend(["--resume", str(s2_resume)])
            run_stage(
                stage_name="stage2_causal_moe",
                cmd=cmd,
                log_path=s2_dir / "logs" / "pipeline_stage2.log",
                completion_marker="Stage 2 Complete",
                expected_ckpt=s2_ckpt_dir / "best_model.pt",
                prune_dir=s2_ckpt_dir,
                prune_pattern="causal_moe_stage2_step_*.pt",
            )

    # ------------------------------------------------------------------
    # Stage 3: Full-Sampling Multi-Source Final Alignment (up to 300k steps + Early Stopping)
    # ------------------------------------------------------------------
    s3_dir = EXPERIMENTS_DIR / "stage3_final_alignment"
    s3_ckpt_dir = s3_dir / "checkpoints"
    if args.from_stage <= 3 <= args.to_stage:
        if is_stage_completed("stage3_final_alignment") and (s3_ckpt_dir / "best_model.pt").exists():
            print("[Pipeline] Stage 3 already completed; skipping.", flush=True)
        else:
            s2_ckpt = resolve_stage_checkpoint(s2_ckpt_dir, "causal_moe_stage2_step_*.pt")
            if not s2_ckpt:
                raise FileNotFoundError(f"Stage 2 checkpoint missing in {s2_ckpt_dir}")
            s3_resume = latest_step_ckpt(s3_ckpt_dir, "causal_moe_stage3_step_*.pt")
            cmd = [
                PYTHON_EXE, "-u", "tasks/run_stage3_alignment.py",
                "--stage2-ckpt", str(s2_ckpt),
                "--steps", str(args.stage3_steps),
                "--batch-size", "32",
                "--grad-accum-steps", "4",
                "--lr", "3e-5",
                "--min-lr", "5e-6",
                "--warmup-steps", "500",
                "--save-every", "1000",
                "--log-every", "25",
                "--keep-last-k", "3",
                "--left-context", str(args.left_context),
                "--early-stop-patience", str(args.early_stop_patience),
            ]
            if s3_resume:
                cmd.extend(["--resume", str(s3_resume)])
            run_stage(
                stage_name="stage3_final_alignment",
                cmd=cmd,
                log_path=s3_dir / "logs" / "pipeline_stage3.log",
                completion_marker="Stage 3 Alignment Complete",
                expected_ckpt=s3_ckpt_dir / "best_model.pt",
                prune_dir=s3_ckpt_dir,
                prune_pattern="causal_moe_stage3_step_*.pt",
            )

    # ------------------------------------------------------------------
    # Stage 4: Full Benchmark Evaluation Suite (RESPIN, KathBath, FLEURS, Streaming RTF)
    # ------------------------------------------------------------------
    s4_dir = EXPERIMENTS_DIR / "e3_full_eval_suite"
    if args.from_stage <= 4 <= args.to_stage:
        if is_stage_completed("e3_full_eval_suite") and (s4_dir / "results" / "summary.json").exists():
            print("[Pipeline] Stage 4 evaluation already completed; skipping.", flush=True)
        else:
            final_ckpt = resolve_stage_checkpoint(s3_ckpt_dir, "causal_moe_stage3_step_*.pt")
            if not final_ckpt:
                raise FileNotFoundError(f"Stage 3 checkpoint missing in {s3_ckpt_dir}")
            cmd = [
                PYTHON_EXE, "-u", "tasks/run_e3_eval.py",
                "--checkpoint", str(final_ckpt),
                "--batch-size", "16",
            ]
            run_stage(
                stage_name="e3_full_eval_suite",
                cmd=cmd,
                log_path=s4_dir / "logs" / "pipeline_eval.log",
                completion_marker="Full evaluation suite completed",
                expected_ckpt=s4_dir / "results" / "summary.json",
            )

    print("\n[Pipeline] ALL STAGES COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
