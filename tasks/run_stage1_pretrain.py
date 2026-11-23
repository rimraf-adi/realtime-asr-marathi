"""
Stage 1: Causal Acoustic Self-Supervised Pretraining Engine for Streaming Conformer.
Aligned with ../marathi-asr:
- Instantiates CausalStreamingASRModel(d_model=512, n_heads=8, ffn_expansion=4, num_layers=12, enable_moe=False)
- Warm-starts from d:/marathi-asr/runs/ablations/stage1/checkpoints/stage1_pretrain/best_model.pt (423 SSL tensors)
- Streams local pretraining shards (d:/marathi-asr/data_cache/pretrain_scaled/chunk_*.pt) asynchronously
  interleaved with local audio manifests (KathBath Train + OpenSLR64) for zero-stall GPU saturation
- Trains for 30,000 steps with masked spectrogram reconstruction (L1 + 0.5*L2), rolling checkpoint pruning, and resume
"""

import os
import gc
import sys
import time
import queue
import random
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARATHI_ASR_ROOT = Path("d:/marathi-asr")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MARATHI_ASR_ROOT) not in sys.path:
    sys.path.insert(1, str(MARATHI_ASR_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.optim import AdamW

import sentinel
from models.causal_conformer.causal_asr_model import CausalStreamingASRModel
from data_utils.kathbath_loader import load_kathbath_manifest
from data_utils.utils import get_lr_scheduler
from pretraining.prefetch_loader import ThreadedPrefetchLoader
from pretraining.metrics import PretrainMetricsTracker


DEFAULT_STAGE1_WARMSTART = Path("d:/marathi-asr/runs/ablations/stage1/checkpoints/stage1_pretrain/best_model.pt")
FALLBACK_OFFLINE_CKPT = Path("d:/realtime/models/marathi-conformer-130m-dense/model.pt")
PRETRAIN_CACHE_DIR = Path("d:/marathi-asr/data_cache/pretrain_scaled")
KATHBATH_MANIFEST = Path("d:/marathi-asr/data/vistaar_benchmarks/kathbath_train/marathi/manifest.json")
KATHBATH_DIR = Path("d:/marathi-asr/data/vistaar_benchmarks/kathbath_train/marathi")
OPENSLR_MANIFEST = Path("d:/marathi-asr/data/openslr64_marathi/manifest.json")
OPENSLR_DIR = Path("d:/marathi-asr/data/openslr64_marathi")


def setup_experiment_dir() -> Dict[str, Path]:
    base_dir = Path("d:/realtime/experiments/stage1_causal_pretrain")
    dirs = {
        "base": base_dir,
        "results": base_dir / "results",
        "plots": base_dir / "plots",
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def log_msg(log_path: Path, msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip(), flush=True)


def prune_step_checkpoints(ckpt_dir: Path, pattern: str = "causal_pretrain_step_*.pt", keep_last_k: int = 3) -> None:
    files = sorted(
        ckpt_dir.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else 0,
    )
    for old_f in files[:-keep_last_k]:
        try:
            old_f.unlink()
        except OSError:
            pass


def find_latest_readable_checkpoint(ckpt_dir: Path, pattern: str = "causal_pretrain_step_*.pt") -> Optional[Path]:
    files = sorted(
        ckpt_dir.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else 0,
        reverse=True,
    )
    for f in files:
        try:
            torch.load(f, map_location="cpu", weights_only=False)
            return f
        except Exception:
            continue
    return None


class AsyncShardPreloader:
    """
    Loads 4.7 GB pretrain_scaled chunk_*.pt shards one ahead in a background thread
    so training never stalls on disk deserialization.
    """

    def __init__(
        self,
        shard_paths: List[Path],
        min_duration: float = 1.5,
        max_duration: float = 14.0,
        seed: int = 42,
    ):
        self.shard_paths = list(shard_paths)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rng = random.Random(seed)
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        if self.shard_paths:
            self.thread = threading.Thread(target=self._producer, daemon=True)
            self.thread.start()

    def _producer(self) -> None:
        while not self.stop_event.is_set():
            order = list(self.shard_paths)
            self.rng.shuffle(order)
            for cp in order:
                if self.stop_event.is_set():
                    break
                try:
                    raw_items = torch.load(cp, map_location="cpu", weights_only=False)
                    valid = []
                    for it in raw_items:
                        dur = float(it.get("duration", 0.0) or 0.0)
                        if self.min_duration <= dur <= self.max_duration and it.get("audio") is not None:
                            valid.append({
                                "audio": it["audio"],
                                "duration": dur,
                                "sampling_rate": int(it.get("sampling_rate", 16000)),
                            })
                    del raw_items
                    gc.collect()
                    self.rng.shuffle(valid)
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put((cp.name, valid), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                except Exception as exc:
                    print(f"[ShardPreloader] Skipping {cp.name}: {exc}", file=sys.stderr, flush=True)

    def try_get_shard(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)


def _load_manifest_audio_item(entry: Dict[str, Any], target_sr: int = 16000) -> Optional[Dict[str, Any]]:
    try:
        arr, sr = sf.read(entry["wav_path"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != target_sr:
            wav_t = torchaudio.functional.resample(torch.from_numpy(arr), sr, target_sr)
            arr = wav_t.numpy()
        dur = len(arr) / float(target_sr)
        if dur < 1.5 or dur > 14.0:
            return None
        return {"audio": arr, "duration": dur, "sampling_rate": target_sr}
    except Exception:
        return None


def build_hybrid_pretrain_stream(
    manifest_entries: List[Dict[str, Any]],
    shard_preloader: AsyncShardPreloader,
    log_path: Path,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """
    Yields pretraining audio dicts by immediately streaming local manifest WAVs
    and interleaving preloaded Shrutilipi/Vaani shards as soon as each shard is ready.
    """
    rng = random.Random(seed)
    manifest_idx = 0
    manifest_pool = list(manifest_entries)
    rng.shuffle(manifest_pool)

    active_shard: List[Dict[str, Any]] = []
    shard_cursor = 0
    turn = 0

    while True:
        # Refill shard buffer non-blockingly when exhausted
        if shard_cursor >= len(active_shard):
            if active_shard:
                active_shard = []
                gc.collect()
            loaded = shard_preloader.try_get_shard()
            if loaded is not None:
                shard_name, active_shard = loaded
                shard_cursor = 0
                log_msg(log_path, f"Streaming preloaded shard {shard_name} ({len(active_shard):,} items).")

        use_shard = (len(active_shard) > 0 and shard_cursor < len(active_shard)) and (
            not manifest_pool or (turn % 2 == 0)
        )
        turn += 1

        if use_shard:
            item = active_shard[shard_cursor]
            shard_cursor += 1
            yield item
        elif manifest_pool:
            if manifest_idx >= len(manifest_pool):
                rng.shuffle(manifest_pool)
                manifest_idx = 0
            entry = manifest_pool[manifest_idx]
            manifest_idx += 1
            loaded_item = _load_manifest_audio_item(entry)
            if loaded_item is not None:
                yield loaded_item


def collate_audio_batch(batch_items: List[Dict[str, Any]]) -> torch.Tensor:
    tensors = [
        torch.from_numpy(item["audio"]).float()
        if isinstance(item["audio"], np.ndarray)
        else item["audio"].float()
        for item in batch_items
    ]
    max_len = max(t.size(0) for t in tensors)
    padded = torch.zeros(len(tensors), max_len, dtype=torch.float32)
    for i, t in enumerate(tensors):
        padded[i, : t.size(0)] = t
    return padded


def run_stage1_pretraining(
    total_steps: int = 300000,
    batch_size: int = 32,
    lr: float = 4e-4,
    warmup_steps: int = 1000,
    save_every: int = 1000,
    log_every: int = 25,
    keep_last_k: int = 3,
    left_context: int = 128,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_steps: int = 10000,
    resume_path: Optional[str] = None,
    warmstart_path: Optional[str] = None,
) -> None:
    dirs = setup_experiment_dir()
    train_log = dirs["logs"] / "train.log"

    log_msg(train_log, "=" * 70)
    log_msg(train_log, "STARTING STAGE 1: CAUSAL ACOUSTIC SELF-SUPERVISED PRETRAINING")
    log_msg(
        train_log,
        f"Max Steps: {total_steps} | Batch Size: {batch_size} | Peak LR: {lr} | Warmup: {warmup_steps} | "
        f"Left Context: {left_context} frames | Early Stop Patience: {early_stop_patience} evals",
    )
    log_msg(train_log, "=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    log_msg(train_log, f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    run = sentinel.init(
        project="causal_conversion_130m",
        run_name="stage1_causal_pretrain",
        config={
            "stage": 1,
            "architecture": "CausalStreamingASRModel (d_model=512, ffn_expansion=4, strictly causal)",
            "left_context": left_context,
            "method": "Masked Spectrogram Reconstruction (SSL)",
            "total_steps": total_steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "warmup_steps": warmup_steps,
            "early_stop_patience": early_stop_patience,
        },
    )

    # 1. Initialize Strictly Causal Streaming Backbone (80.76M encoder+frontend+recon, bounded left context)
    model = CausalStreamingASRModel(
        d_model=512,
        n_heads=8,
        ffn_expansion=4,
        num_layers=12,
        conv_kernel_size=31,
        dropout=0.1,
        left_context=left_context,
        enable_moe=False,
    ).to(device)

    # Causality Gate (Test A: zero future lookahead verification)
    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 100, 80, device=device)
        x2 = x1.clone()
        x2[:, 50:, :] = torch.randn(1, 50, 80, device=device)
        h1 = model.forward(x1, is_waveform=False)["final_hidden"]
        h2 = model.forward(x2, is_waveform=False)["final_hidden"]
        prefix_diff = (h1[:, :12, :] - h2[:, :12, :]).abs().max().item()
        assert prefix_diff < 1e-6, f"CAUSALITY VIOLATION: future perturbation leaked {prefix_diff}"
    log_msg(train_log, f"[Causality Gate PASSED] Future perturbation prefix diff = {prefix_diff:.2e} (left_context={left_context})")

    # Freeze unused CTC head during Stage 1 SSL
    for p in model.ctc_head.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log_msg(train_log, f"Stage 1 trainable parameters (encoder+recon): {sum(p.numel() for p in trainable_params):,}")

    # Train from scratch by default (strictly causal real-time representations); optional explicit --warmstart
    if not resume_path and warmstart_path:
        start_ckpt = Path(warmstart_path)
        if start_ckpt.exists():
            res = model.load_offline_checkpoint(str(start_ckpt), strict=False)
            log_msg(train_log, f"Warm-started {len(res['loaded_keys'])} tensors from {start_ckpt}")
    elif not resume_path:
        log_msg(train_log, "Training strictly causal Stage 1 backbone FROM SCRATCH (no offline bidirectional warm-start).")

    # 2. Optimizer & Scheduler
    optimizer = AdamW(trainable_params, lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.05)

    step = 0
    total_audio_sec = 0.0
    best_masked_l1 = float("inf")
    early_stop_counter = 0

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        step = int(ckpt.get("step", 0))
        total_audio_sec = float(ckpt.get("total_audio_hours", 0.0)) * 3600.0
        best_masked_l1 = float(ckpt.get("best_masked_l1", ckpt.get("masked_l1", float("inf"))))
        early_stop_counter = int(ckpt.get("early_stop_counter", 0))
        log_msg(
            train_log,
            f"Resumed Stage 1 from {resume_path} at step {step} "
            f"(best L1: {best_masked_l1:.4f}, patience: {early_stop_counter}/{early_stop_patience})",
        )

    if step >= total_steps:
        log_msg(train_log, f"Stage 1 already at step {step} >= {total_steps}. Skipping training.")
        run.finish()
        return

    # 3. Hybrid Local Manifest + Async Pretrain Shard Stream
    manifest_entries: List[Dict[str, Any]] = []
    if KATHBATH_MANIFEST.exists():
        manifest_entries.extend(load_kathbath_manifest(str(KATHBATH_MANIFEST), str(KATHBATH_DIR)))
    if OPENSLR_MANIFEST.exists():
        manifest_entries.extend(load_kathbath_manifest(str(OPENSLR_MANIFEST), str(OPENSLR_DIR)))
    manifest_entries = [e for e in manifest_entries if 1.5 <= float(e.get("duration", 0.0) or 0.0) <= 14.0]

    shard_paths = sorted(PRETRAIN_CACHE_DIR.glob("chunk_*.pt")) if PRETRAIN_CACHE_DIR.exists() else []
    log_msg(
        train_log,
        f"Data sources ready: {len(shard_paths)} local shards (~{len(shard_paths)*10000:,} samples) "
        f"+ {len(manifest_entries):,} local manifest utterances.",
    )

    shard_preloader = AsyncShardPreloader(shard_paths, min_duration=1.5, max_duration=14.0, seed=42)
    stream = build_hybrid_pretrain_stream(manifest_entries, shard_preloader, train_log, seed=42)
    prefetcher = ThreadedPrefetchLoader(
        stream_iterator=stream,
        batch_size=batch_size,
        max_prefetch=8,
        collate_fn=collate_audio_batch,
        device=device,
        min_duration=1.5,
        max_duration=14.0,
        infinite=False,
    )

    model.train()
    start_time = time.perf_counter()
    rolling_l1 = 0.0
    rolling_snr = 0.0

    try:
        for waveforms, raw_items in prefetcher:
            if step >= total_steps:
                break

            step += 1
            waveforms = waveforms.to(device, non_blocking=True)
            batch_dur = sum(float(x.get("duration", 0.0)) for x in raw_items)
            total_audio_sec += batch_dur

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = model.forward_pretrain(waveforms)
                    orig = outputs["original_spec"]
                    recon = outputs["reconstructed_spec"]
                    mask = outputs["mask"]

                    mask_3d = mask.unsqueeze(-1).expand_as(orig)
                    if mask_3d.sum() > 0:
                        loss_l1 = torch.abs(orig[mask_3d] - recon[mask_3d]).mean()
                        loss_l2 = torch.mean((orig[mask_3d] - recon[mask_3d]) ** 2)
                        loss = loss_l1 + 0.5 * loss_l2
                    else:
                        loss = torch.tensor(0.0, device=orig.device, requires_grad=True)

                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0).item()
                optimizer.step()
                scheduler.step()
            except torch.OutOfMemoryError:
                log_msg(train_log, f"[OOM Warning] Skipped step {step} due to CUDA OOM.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            with torch.no_grad():
                recon_metrics = PretrainMetricsTracker.compute_reconstruction_metrics(orig.float(), recon.float(), mask)
            l1_val = recon_metrics.get("loss_masked_l1", 0.0)
            snr_val = recon_metrics.get("reconstruction_snr_db", 0.0)
            rolling_l1 = 0.9 * rolling_l1 + 0.1 * l1_val if rolling_l1 > 0 else l1_val
            rolling_snr = 0.9 * rolling_snr + 0.1 * snr_val if rolling_snr != 0 else snr_val
            current_lr = scheduler.get_last_lr()[0]

            if step % log_every == 0 or step == 1:
                log_msg(
                    train_log,
                    f"Step {step:05d}/{total_steps} | Loss: {loss.item():.4f} "
                    f"(L1: {l1_val:.4f}, RollL1: {rolling_l1:.4f}, SNR: {snr_val:.2f} dB) | "
                    f"GradNorm: {grad_norm:.2f} | Audio: {total_audio_sec/3600:.2f} hrs | LR: {current_lr:.2e}",
                )
                run.log({
                    "pretrain/loss": float(loss.item()),
                    "pretrain/masked_l1": l1_val,
                    "pretrain/rolling_l1": rolling_l1,
                    "pretrain/snr_db": snr_val,
                    "pretrain/grad_norm": grad_norm,
                    "pretrain/lr": current_lr,
                    "pretrain/audio_hours": total_audio_sec / 3600.0,
                }, step=step)

            if step % save_every == 0 or step == total_steps:
                ckpt_payload = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "masked_l1": rolling_l1,
                    "snr_db": rolling_snr,
                    "total_audio_hours": total_audio_sec / 3600.0,
                    "best_masked_l1": min(best_masked_l1, rolling_l1),
                    "model_config": {
                        "d_model": 512,
                        "n_heads": 8,
                        "ffn_expansion": 4,
                        "num_layers": 12,
                        "vocab_size": 105,
                    },
                }
                ckpt_path = dirs["checkpoints"] / f"causal_pretrain_step_{step}.pt"
                torch.save(ckpt_payload, ckpt_path)
                log_msg(train_log, f"Checkpoint saved at step {step} -> {ckpt_path}")

                improved = rolling_l1 < (best_masked_l1 - early_stop_min_delta)
                if improved or not (dirs["checkpoints"] / "best_model.pt").exists():
                    best_masked_l1 = min(best_masked_l1, rolling_l1)
                    early_stop_counter = 0
                    ckpt_payload["best_masked_l1"] = best_masked_l1
                    ckpt_payload["early_stop_counter"] = early_stop_counter
                    best_path = dirs["checkpoints"] / "best_model.pt"
                    torch.save(ckpt_payload, best_path)
                    log_msg(train_log, f"Updated best Stage 1 checkpoint (rolling L1={best_masked_l1:.4f}) -> {best_path}")
                elif step >= early_stop_min_steps:
                    early_stop_counter += 1
                    ckpt_payload["early_stop_counter"] = early_stop_counter
                    torch.save(ckpt_payload, ckpt_path)
                    log_msg(
                        train_log,
                        f"[Early Stop] No L1 improvement ({early_stop_counter}/{early_stop_patience}; "
                        f"current={rolling_l1:.4f}, best={best_masked_l1:.4f})",
                    )
                    if early_stop_counter >= early_stop_patience:
                        log_msg(train_log, f"[Early Stop] Triggered at step {step}! Stopping Stage 1 pretraining cleanly.")
                        prune_step_checkpoints(dirs["checkpoints"], "causal_pretrain_step_*.pt", keep_last_k=keep_last_k)
                        break

                prune_step_checkpoints(dirs["checkpoints"], "causal_pretrain_step_*.pt", keep_last_k=keep_last_k)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        prefetcher.close()
        shard_preloader.close()
        run.finish()

    elapsed = time.perf_counter() - start_time
    log_msg(train_log, f"Stage 1 pretraining completed in {elapsed/3600:.2f} hrs ({total_audio_sec/3600:.1f} hrs audio processed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Causal Acoustic Self-Supervised Pretraining")
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-last-k", type=int, default=3)
    parser.add_argument("--left-context", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-min-steps", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--warmstart", type=str, default=None)
    args = parser.parse_args()

    run_stage1_pretraining(
        total_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        keep_last_k=args.keep_last_k,
        left_context=args.left_context,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_steps=args.early_stop_min_steps,
        resume_path=args.resume,
        warmstart_path=args.warmstart,
    )
