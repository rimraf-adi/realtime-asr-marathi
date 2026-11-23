"""
Stage 3: Multi-Source Full-Sampling Final Alignment & Calibration Engine for Causal MoE Conformer.
- Strictly causal real-time streaming architecture (left_context=128 frames = 5.12s, zero future lookahead).
- Loads Stage 2 best checkpoint into 130.8M CausalStreamingASRModel.
- Freezes frontend, subsampling, pos_enc, and Encoder Layers 1-4; trains Layers 5-12 + CTC head (75.7%).
- Full uniform sampling (NO artificial corpus upweighting) with length-bucketed epoch passes across all 5 corpora:
    1. RESPIN (Train + Held-out Calib minus 128-utt early-stopping val slice)
    2. OpenSLR64 Marathi (1,569 utts)
    3. AI4Bharat KathBath Marathi (Clean 84,070 + Noisy 1,631 = 85,701)
    4. Common Voice Marathi (1,205 utts)
    5. Transcribed Local Pretraining Shards (~50,000 utts at natural corpus proportion)
- Trains up to 300,000 steps with early stopping patience on held-out speaker-disjoint CER.
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
from typing import List, Dict, Any, Optional

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import sentinel
from models.causal_conformer.causal_asr_model import CausalStreamingASRModel
from model.masking import SpecAugment
from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer, get_lr_scheduler
from data_utils.kathbath_loader import load_kathbath_manifest
from data_utils.multi_source_calib_loader import collate_multi_source_batch
from moe.respin_dataset import build_or_load_speaker_split
from tasks.run_stage2_moe_adapt import evaluate_heldout_cer


DEFAULT_STAGE2_CKPT = Path("d:/realtime/experiments/stage2_causal_moe/checkpoints/best_model.pt")
DEFAULT_TEACHER_CKPT = Path("d:/marathi-asr/runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt")
VOCAB_PATH = Path("d:/realtime/data_utils/vocab.json")
RESPIN_CACHE_PATH = Path("d:/marathi-asr/moe/respin_split_cache.json")
PRETRAIN_CACHE_DIR = Path("d:/marathi-asr/data_cache/pretrain_scaled")


def setup_experiment_dir() -> Dict[str, Path]:
    base_dir = Path("d:/realtime/experiments/stage3_final_alignment")
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


def prune_step_checkpoints(ckpt_dir: Path, pattern: str = "causal_moe_stage3_step_*.pt", keep_last_k: int = 3) -> None:
    files = sorted(
        ckpt_dir.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else 0,
    )
    for old_f in files[:-keep_last_k]:
        try:
            old_f.unlink()
        except OSError:
            pass


class FullPoolMultiSourceLoader:
    """
    Full uniform (unweighted) epoch-based prefetch loader across all 5 local Marathi corpora.
    Uses length-bucketing sort across the unified shuffled pool for high GPU throughput,
    and mixes in transcribed pretrain_scaled shards at their natural size proportion.
    """

    def __init__(
        self,
        tokenizer: MarathiTokenizer,
        batch_size: int = 32,
        queue_size: int = 6,
        bucket_multiplier: int = 4,
        augment: bool = True,
        log_path: Optional[Path] = None,
        non_respin_only: bool = False,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.bucket_multiplier = bucket_multiplier
        self.augment = augment
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.shard_queue: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.log_path = log_path

        # 1. RESPIN (Hold out 128 utterances from 5% calib for early stopping validation)
        train_utts, calib_utts = build_or_load_speaker_split(cache_path=str(RESPIN_CACHE_PATH))
        self.val_subset = calib_utts[:128]
        respin_train_pool = train_utts + calib_utts[128:]

        # 2. OpenSLR64 Marathi
        slr_manifest = Path("d:/marathi-asr/data/openslr64_marathi/manifest.json")
        slr_dir = Path("d:/marathi-asr/data/openslr64_marathi")
        openslr_pool = load_kathbath_manifest(str(slr_manifest), str(slr_dir)) if slr_manifest.exists() else []
        for it in openslr_pool:
            it["dialect_idx"] = 3

        # 3. KathBath Clean Train + Noisy
        kathbath_pool: List[Dict[str, Any]] = []
        kb_train = Path("d:/marathi-asr/data/vistaar_benchmarks/kathbath_train/marathi/manifest.json")
        kb_train_dir = Path("d:/marathi-asr/data/vistaar_benchmarks/kathbath_train/marathi")
        kb_noisy = Path("d:/marathi-asr/data/vistaar_benchmarks/kathbath_noisy/marathi/manifest.json")
        kb_noisy_dir = Path("d:/marathi-asr/data/vistaar_benchmarks/kathbath_noisy/marathi")
        if kb_train.exists():
            kathbath_pool.extend(load_kathbath_manifest(str(kb_train), str(kb_train_dir)))
        if kb_noisy.exists():
            kathbath_pool.extend(load_kathbath_manifest(str(kb_noisy), str(kb_noisy_dir)))
        for it in kathbath_pool:
            it["dialect_idx"] = 3

        # 4. Common Voice Marathi
        cv_manifest = Path("d:/marathi-asr/data/vistaar_benchmarks/commonvoice/marathi/manifest.json")
        cv_dir = Path("d:/marathi-asr/data/vistaar_benchmarks/commonvoice/marathi")
        cv_pool = load_kathbath_manifest(str(cv_manifest), str(cv_dir)) if cv_manifest.exists() else []
        for it in cv_pool:
            it["dialect_idx"] = 3

        # Unified unweighted full local pool
        if non_respin_only:
            self.val_subset = []
            self.full_local_pool = openslr_pool + kathbath_pool + cv_pool
            self.shard_paths = []
            self.natural_shard_prob = 0.0
            est_shard_items = 0
        else:
            self.full_local_pool = respin_train_pool + openslr_pool + kathbath_pool + cv_pool
            # 5. Pretrain shard paths (~5,000 transcribed items per shard -> natural proportion ~50k / (900k + 50k))
            self.shard_paths = sorted(PRETRAIN_CACHE_DIR.glob("chunk_*.pt")) if PRETRAIN_CACHE_DIR.exists() else []
            est_shard_items = len(self.shard_paths) * 5000
            self.natural_shard_prob = est_shard_items / max(1, len(self.full_local_pool) + est_shard_items)

        summary = (
            f"Full Uniform Pool Ready (NO artificial weighting): {len(self.full_local_pool):,} local utts "
            f"(RESPIN: {len(respin_train_pool):,}, OpenSLR64: {len(openslr_pool):,}, "
            f"KathBath: {len(kathbath_pool):,}, CommonVoice: {len(cv_pool):,}) + "
            f"{len(self.shard_paths)} shards (~{est_shard_items:,} transcribed utts, natural p={self.natural_shard_prob:.3f}) | "
            f"Held-out val: {len(self.val_subset)} utts."
        )
        if self.log_path is not None:
            log_msg(self.log_path, summary)
        else:
            print(summary, flush=True)

        self.active_shard: List[Dict[str, Any]] = []
        self.shard_cursor = 0

        self.shard_thread: Optional[threading.Thread] = None
        if self.shard_paths:
            self.shard_thread = threading.Thread(target=self._shard_producer, daemon=True)
            self.shard_thread.start()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _shard_producer(self) -> None:
        rng = random.Random(101)
        while not self.stop_event.is_set():
            order = list(self.shard_paths)
            rng.shuffle(order)
            for cp in order:
                if self.stop_event.is_set():
                    break
                try:
                    raw = torch.load(cp, map_location="cpu", weights_only=False)
                    transcribed = []
                    for it in raw:
                        dur = float(it.get("duration", 0.0) or 0.0)
                        if not (0.8 <= dur <= 15.0) or it.get("audio") is None:
                            continue
                        txt = sanitize_transcript(it.get("text", ""))
                        if len(txt) > 2:
                            transcribed.append({
                                "audio": it["audio"],
                                "sampling_rate": int(it.get("sampling_rate", 16000)),
                                "duration": dur,
                                "text": txt,
                                "dialect_idx": 3,
                            })
                    del raw
                    gc.collect()
                    rng.shuffle(transcribed)
                    while not self.stop_event.is_set():
                        try:
                            self.shard_queue.put(transcribed, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                except Exception:
                    continue

    def _try_get_shard_item(self) -> Optional[Dict[str, Any]]:
        if self.shard_cursor >= len(self.active_shard):
            if self.active_shard:
                self.active_shard = []
                gc.collect()
            try:
                self.active_shard = self.shard_queue.get_nowait()
                self.shard_cursor = 0
            except queue.Empty:
                return None

        if self.shard_cursor < len(self.active_shard):
            item = self.active_shard[self.shard_cursor]
            self.shard_cursor += 1
            return item
        return None

    def _worker_loop(self) -> None:
        pool = list(self.full_local_pool)
        bucket_size = self.batch_size * self.bucket_multiplier

        while not self.stop_event.is_set():
            random.shuffle(pool)
            for i in range(0, len(pool), bucket_size):
                if self.stop_event.is_set():
                    break
                chunk = list(pool[i : i + bucket_size])
                # Mix in transcribed shard samples at natural corpus proportion
                if self.natural_shard_prob > 0:
                    for k in range(len(chunk)):
                        if random.random() < self.natural_shard_prob:
                            shard_it = self._try_get_shard_item()
                            if shard_it is not None:
                                chunk[k] = shard_it

                # Length-bucketing sort to eliminate padding waste
                chunk.sort(key=lambda x: float(x.get("duration", 0.0) or 0.0))

                for j in range(0, len(chunk), self.batch_size):
                    if self.stop_event.is_set():
                        break
                    batch_samples = chunk[j : j + self.batch_size]
                    batch = collate_multi_source_batch(batch_samples, self.tokenizer, augment=self.augment)
                    if batch is not None:
                        while not self.stop_event.is_set():
                            try:
                                self.queue.put(batch, timeout=0.5)
                                break
                            except queue.Full:
                                continue

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        return self.queue.get()

    def close(self) -> None:
        self.stop_event.set()
        for q in (self.queue, self.shard_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        if self.shard_thread is not None and self.shard_thread.is_alive():
            self.shard_thread.join(timeout=2.0)


def run_stage3_alignment(
    stage2_ckpt_path: Optional[str] = None,
    total_steps: int = 300000,
    batch_size: int = 32,
    grad_accum_steps: int = 4,
    lr: float = 3e-5,
    min_lr: float = 5e-6,
    warmup_steps: int = 500,
    save_every: int = 1000,
    log_every: int = 25,
    keep_last_k: int = 3,
    left_context: int = 128,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 0.001,
    early_stop_min_steps: int = 5000,
    resume_path: Optional[str] = None,
    non_respin_only: bool = False,
    exp_name: str = "stage3_final_alignment",
) -> None:
    global EXP_DIR
    EXP_DIR = PROJECT_ROOT / "experiments" / exp_name
    dirs = setup_experiment_dir()
    train_log = dirs["logs"] / "train.log"

    s2_ckpt = stage2_ckpt_path or str(DEFAULT_STAGE2_CKPT)
    if not os.path.exists(s2_ckpt):
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {s2_ckpt}")

    log_msg(train_log, "=" * 70)
    log_msg(train_log, "STARTING STAGE 3: FULL-SAMPLING CAUSAL FINAL ALIGNMENT (PURE CTC)")
    log_msg(
        train_log,
        f"Max Steps: {total_steps} | Batch: {batch_size} x Accum {grad_accum_steps} | "
        f"LR: {lr} -> {min_lr} | Left Context: {left_context} | Early Stop Patience: {early_stop_patience} evals",
    )
    log_msg(train_log, "=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    log_msg(train_log, f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    run = sentinel.init(
        project="causal_conversion_130m",
        run_name="stage3_final_alignment",
        config={
            "stage": 3,
            "architecture": "CausalStreamingASRModel (130M MoE, strictly causal, from scratch)",
            "left_context": left_context,
            "sampling": "full_uniform_unweighted",
            "objective": "pure_ctc",
            "total_steps": total_steps,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "lr": lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "early_stop_patience": early_stop_patience,
            "stage2_checkpoint": s2_ckpt,
        },
    )

    # 1. Load Stage 2 Strictly Causal MoE Model
    model = CausalStreamingASRModel(
        d_model=512,
        num_layers=12,
        n_heads=8,
        conv_kernel_size=31,
        ffn_expansion=4,
        enable_moe=True,
        num_experts=3,
        moe_start_layer=5,
        dropout=0.1,
        left_context=left_context,
        vocab_size=105,
    ).to(device)

    ckpt_data = torch.load(s2_ckpt, map_location=device, weights_only=False)
    sd = ckpt_data["model_state_dict"] if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(sd, strict=True)
    log_msg(train_log, f"Loaded Stage 2 weights from {s2_ckpt}")

    # Causality Gate (Test A: zero future lookahead verification)
    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 100, 80, device=device)
        x2 = x1.clone()
        x2[:, 50:, :] = torch.randn(1, 50, 80, device=device)
        h1 = model.forward(x1, is_waveform=False)["final_hidden"]
        h2 = model.forward(x2, is_waveform=False)["final_hidden"]
        prefix_diff = (h1[:, :12, :] - h2[:, :12, :]).abs().max().item()
        assert prefix_diff < 1e-5, f"CAUSALITY VIOLATION: future perturbation leaked {prefix_diff}"
    log_msg(train_log, f"[Causality Gate PASSED] Future perturbation prefix diff = {prefix_diff:.2e} (left_context={left_context})")

    # 2. Freeze frontend, reconstruction_head, subsampling, pos_enc, and Layers 1-4; train Layers 5-12 + CTC
    for p in model.parameters():
        p.requires_grad = False

    trainable_params: List[nn.Parameter] = []
    for layer_idx in range(4, 12):
        for p in model.encoder.layers[layer_idx].parameters():
            p.requires_grad = True
            trainable_params.append(p)
    for p in model.ctc_head.parameters():
        p.requires_grad = True
        trainable_params.append(p)

    asr_total_params = sum(
        p.numel() for n, p in model.named_parameters() if not n.startswith("reconstruction_head.")
    )
    num_trainable = sum(p.numel() for p in trainable_params)
    log_msg(
        train_log,
        f"Freezing Layers 1-4. Training Layers 5-12 + CTC: {num_trainable:,} / {asr_total_params:,} params "
        f"({100.0 * num_trainable / asr_total_params:.1f}%)",
    )

    # 3. Optimizer & Cosine Scheduler
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-8)
    total_opt_steps = max(1, total_steps // grad_accum_steps)
    warmup_opt_steps = max(1, warmup_steps // grad_accum_steps)
    min_lr_ratio = max(0.01, min_lr / max(lr, 1e-9))
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=warmup_opt_steps,
        total_steps=total_opt_steps,
        min_lr_ratio=min_lr_ratio,
    )

    step = 0
    rolling_loss = 0.0
    rolling_cer = 1.0
    best_cer = float("inf")
    early_stop_counter = 0

    if resume_path and os.path.exists(resume_path):
        r_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(r_ckpt["model_state_dict"])
        if "optimizer_state_dict" in r_ckpt:
            optimizer.load_state_dict(r_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in r_ckpt:
            scheduler.load_state_dict(r_ckpt["scheduler_state_dict"])
        step = int(r_ckpt.get("step", 0))
        rolling_loss = float(r_ckpt.get("ctc_loss", 0.0))
        best_cer = float(r_ckpt.get("best_cer", r_ckpt.get("val_cer", r_ckpt.get("cer", float("inf")))))
        rolling_cer = float(r_ckpt.get("cer", 1.0))
        early_stop_counter = int(r_ckpt.get("early_stop_counter", 0))
        log_msg(
            train_log,
            f"Resumed Stage 3 from {resume_path} at step {step} "
            f"(best val CER: {best_cer:.4f}, patience: {early_stop_counter}/{early_stop_patience})",
        )

    if step >= total_steps:
        log_msg(train_log, f"Stage 3 already at step {step} >= {total_steps}. Skipping training.")
        run.finish()
        return

    # 4. Full Uniform Multi-Source Data Loader
    tokenizer = MarathiTokenizer(str(VOCAB_PATH))
    loader = FullPoolMultiSourceLoader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        queue_size=6,
        bucket_multiplier=4,
        augment=True,
        log_path=train_log,
    )
    val_subset = loader.val_subset

    ctc_criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    spec_aug = SpecAugment(freq_mask_max=27, time_mask_max=40, num_freq_masks=2, num_time_masks=5).to(device)

    model.train()
    spec_aug.train()
    start_time = time.perf_counter()
    grad_norm_val = 0.0
    last_cer = rolling_cer

    try:
        while step < total_steps:
            step += 1
            step_start = time.perf_counter()
            batch = next(loader)

            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = batch["audio_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lens = batch["target_lengths"].to(device, non_blocking=True)

            if (step - 1) % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    with torch.no_grad():
                        spec = model.frontend(audio)
                        spec = spec_aug(spec)

                    outputs = model(spec, is_waveform=False, dialect_idx=None, use_hard_routing=False)
                    logits = outputs["logits"]
                    log_probs = F.log_softmax(logits, dim=-1)
                    log_probs_t = log_probs.transpose(0, 1)

                    input_lengths = torch.clamp(audio_lens // (160 * 4), min=1, max=logits.size(1))
                    total_loss = ctc_criterion(log_probs_t, targets, input_lengths, target_lens)

                (total_loss / grad_accum_steps).backward()
            except torch.OutOfMemoryError:
                log_msg(train_log, f"[OOM Warning] Skipped step {step} due to CUDA OOM.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            if step % grad_accum_steps == 0:
                grad_norm = nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                grad_norm_val = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step_time = time.perf_counter() - step_start
            batch_audio_sec = float(audio_lens.sum().item()) / 16000.0
            throughput = batch_audio_sec / max(step_time, 1e-5)

            loss_val = float(total_loss.item())
            rolling_loss = 0.9 * rolling_loss + 0.1 * loss_val if rolling_loss > 0 else loss_val

            if step % log_every == 0 or step == 1:
                with torch.no_grad():
                    eval_n = min(8, audio.size(0))
                    batch_cers = []
                    for s_i in range(eval_n):
                        valid_t = int(input_lengths[s_i].item())
                        p_toks = log_probs[s_i, :valid_t].argmax(dim=-1).tolist()
                        p_txt = tokenizer.ctc_decode(p_toks)
                        r_txt = batch["texts"][s_i]
                        batch_cers.append(compute_cer(r_txt, p_txt))
                    last_cer = sum(batch_cers) / max(len(batch_cers), 1)
                    rolling_cer = 0.9 * rolling_cer + 0.1 * last_cer if step > 1 else last_cer
                    ref_text = batch["texts"][0]
                    pred_text = tokenizer.ctc_decode(
                        log_probs[0, : int(input_lengths[0].item())].argmax(dim=-1).tolist()
                    )

                curr_lr = scheduler.get_last_lr()[0]
                log_msg(
                    train_log,
                    f"Step {step:6d}/{total_steps} | Stage 3 Align | "
                    f"CTC Loss: {rolling_loss:.4f} | "
                    f"CER: {last_cer:.3f} (Roll: {rolling_cer:.3f}) | "
                    f"GradNorm: {grad_norm_val:.2f} | Thru: {throughput:5.1f} aud-s/s | LR: {curr_lr:.2e}",
                )
                log_msg(train_log, f"  [Ref ]: {ref_text[:80]}")
                log_msg(train_log, f"  [Pred]: {pred_text[:80]}")

                run.log({
                    "ctc_loss": rolling_loss,
                    "cer": last_cer,
                    "rolling_cer": rolling_cer,
                    "grad_norm": grad_norm_val,
                    "throughput_aud_s_per_s": throughput,
                    "lr": curr_lr,
                }, step=step)

            if step % save_every == 0 or step == total_steps:
                val_cer = evaluate_heldout_cer(model, val_subset, tokenizer, device, batch_size=16)
                log_msg(train_log, f"[Validation @ Step {step}] Held-out Speaker-Disjoint CER: {val_cer:.4f} (best: {best_cer:.4f})")
                run.log({"val_cer": val_cer}, step=step)

                improved = val_cer < (best_cer - early_stop_min_delta)
                ckpt_payload = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "ctc_loss": rolling_loss,
                    "cer": rolling_cer,
                    "val_cer": val_cer,
                    "best_cer": min(best_cer, val_cer),
                    "early_stop_counter": early_stop_counter,
                    "model_config": {
                        "d_model": 512,
                        "n_heads": 8,
                        "ffn_expansion": 4,
                        "num_layers": 12,
                        "left_context": left_context,
                        "vocab_size": 105,
                    },
                }
                ckpt_path = dirs["checkpoints"] / f"causal_moe_stage3_step_{step}.pt"
                best_path = dirs["checkpoints"] / "best_model.pt"

                if improved or not best_path.exists():
                    best_cer = min(best_cer, val_cer)
                    early_stop_counter = 0
                    ckpt_payload["best_cer"] = best_cer
                    ckpt_payload["early_stop_counter"] = early_stop_counter
                    torch.save(ckpt_payload, ckpt_path)
                    torch.save(ckpt_payload, best_path)
                    log_msg(train_log, f"Updated best Stage 3 checkpoint (val CER={best_cer:.4f}) -> {best_path}")
                else:
                    if step >= early_stop_min_steps:
                        early_stop_counter += 1
                    ckpt_payload["early_stop_counter"] = early_stop_counter
                    torch.save(ckpt_payload, ckpt_path)
                    log_msg(
                        train_log,
                        f"Checkpoint saved at: {ckpt_path} | Early Stop Patience: {early_stop_counter}/{early_stop_patience}",
                    )
                    if step >= early_stop_min_steps and early_stop_counter >= early_stop_patience:
                        log_msg(train_log, f"[Early Stop] Triggered at step {step}! Stopping Stage 3 cleanly.")
                        prune_step_checkpoints(dirs["checkpoints"], "causal_moe_stage3_step_*.pt", keep_last_k=keep_last_k)
                        break

                prune_step_checkpoints(dirs["checkpoints"], "causal_moe_stage3_step_*.pt", keep_last_k=keep_last_k)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        loader.close()
        run.finish()

    elapsed_hrs = (time.perf_counter() - start_time) / 3600.0
    log_msg(train_log, f"Stage 3 Alignment Complete in {elapsed_hrs:.2f} hrs! Best checkpoint: {dirs['checkpoints'] / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Full-Sampling Multi-Source Final Alignment")
    parser.add_argument("--stage2-ckpt", type=str, default=str(DEFAULT_STAGE2_CKPT))
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-last-k", type=int, default=3)
    parser.add_argument("--left-context", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-min-steps", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--non-respin-only", action="store_true", help="Train explicitly on KathBath, OpenSLR64, and CommonVoice (no RESPIN/shards)")
    parser.add_argument("--exp-name", type=str, default="stage3_final_alignment")
    args = parser.parse_args()

    run_stage3_alignment(
        stage2_ckpt_path=args.stage2_ckpt,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        keep_last_k=args.keep_last_k,
        left_context=args.left_context,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_steps=args.early_stop_min_steps,
        resume_path=args.resume,
        non_respin_only=args.non_respin_only,
        exp_name=args.exp_name,
    )
