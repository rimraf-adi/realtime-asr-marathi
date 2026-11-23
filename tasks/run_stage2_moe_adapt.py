"""
Stage 2: Causal MoE Dialect Adaptation Engine.
Aligned with ../marathi-asr:
- Initializes from Stage 1 causal pretrained backbone (ffn_expansion=4, d_model=512).
- Complete sparse upcycling of encoder blocks 5-12 FFN2 weights into BOTH combining_ffn
  and all 3 regional dialect experts (D1: Malvani, D2: Ahirani, D4: Varhadi), totaling 130.8M params.
- Differential learning rates: frontend frozen, layers 1-4 + subsampling at 0.1x LR, layers 5-12 + CTC at 1.0x LR.
- Speaker-disjoint RESPIN training split (d:/marathi-asr/moe/respin_split_cache.json) via length-bucketed LocalRESPINPrefetchLoader.
- 3-phase schedule (default 80,000 steps / ~3.33 epochs):
    Phase 0  (Steps 1 - 3,000):        CTC warmup (pure CTC loss)
    Phase 4A (Steps 3,001 - 18,000):   Metadata-guided hard routing + aux load balancing
    Phase 4B (Steps 18,001 - 80,000):  Autonomous frame-wise top-1 routing + aux load balancing
"""

import os
import sys
import time
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
from models.causal_conformer.moe import SparseMoELayer
from model.masking import SpecAugment
from data_utils.tokenizer import MarathiTokenizer
from data_utils.utils import compute_cer, get_lr_scheduler
from moe.respin_dataset import build_or_load_speaker_split, LocalRESPINPrefetchLoader


DEFAULT_STAGE1_CKPT = Path("d:/realtime/experiments/stage1_causal_pretrain/checkpoints/best_model.pt")
VOCAB_PATH = Path("d:/realtime/data_utils/vocab.json")
RESPIN_CACHE_PATH = Path("d:/marathi-asr/moe/respin_split_cache.json")


def setup_experiment_dir() -> Dict[str, Path]:
    base_dir = Path("d:/realtime/experiments/stage2_causal_moe")
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


def prune_step_checkpoints(ckpt_dir: Path, pattern: str = "causal_moe_stage2_step_*.pt", keep_last_k: int = 3) -> None:
    files = sorted(
        ckpt_dir.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else 0,
    )
    for old_f in files[:-keep_last_k]:
        try:
            old_f.unlink()
        except OSError:
            pass


def upcycle_stage1_to_moe(
    stage1_path: str,
    vocab_size: int = 105,
    num_experts: int = 3,
    moe_start_layer: int = 5,
    left_context: int = 128,
    log_path: Optional[Path] = None,
) -> CausalStreamingASRModel:
    """
    Instantiate 130.8M Causal MoE model and perform function-preserving sparse upcycling:
    clones dense FFN2 weights in layers 5-12 into BOTH combining_ffn AND all 3 experts.
    """
    moe_model = CausalStreamingASRModel(
        d_model=512,
        num_layers=12,
        n_heads=8,
        conv_kernel_size=31,
        ffn_expansion=4,
        enable_moe=True,
        num_experts=num_experts,
        moe_start_layer=moe_start_layer,
        dropout=0.1,
        left_context=left_context,
        vocab_size=vocab_size,
    )

    dense_ckpt = torch.load(stage1_path, map_location="cpu", weights_only=False)
    raw_sd = dense_ckpt["model_state_dict"] if isinstance(dense_ckpt, dict) and "model_state_dict" in dense_ckpt else dense_ckpt
    dense_sd = {(k[len("backbone."):] if k.startswith("backbone.") else k): v for k, v in raw_sd.items()}

    sub_aliases = {
        "encoder.subsampling.conv1.weight": "encoder.subsampling.conv.0.weight",
        "encoder.subsampling.conv1.bias": "encoder.subsampling.conv.0.bias",
        "encoder.subsampling.conv2.weight": "encoder.subsampling.conv.2.weight",
        "encoder.subsampling.conv2.bias": "encoder.subsampling.conv.2.bias",
    }

    moe_sd = moe_model.state_dict()
    transferred = 0
    combining_cloned = 0
    experts_cloned = 0
    populated_keys = set()

    with torch.no_grad():
        # 1. Direct backbone parameter transfer (subsampling, pos_enc, layers 1-4 all, layers 5-12 attn/conv/ffn1/norm)
        for k, target_t in moe_sd.items():
            src_k = k if k in dense_sd else sub_aliases.get(k)
            if src_k and src_k in dense_sd and dense_sd[src_k].shape == target_t.shape:
                target_t.copy_(dense_sd[src_k])
                transferred += 1
                populated_keys.add(k)

        # 2. Sparse upcycling of FFN2 for layers moe_start_layer-1 .. 11 into combining_ffn + all experts
        for layer_idx in range(moe_start_layer - 1, 12):
            prefix = f"encoder.layers.{layer_idx}.ffn2."
            for suffix in (
                "layer_norm.weight",
                "layer_norm.bias",
                "linear1.weight",
                "linear1.bias",
                "linear2.weight",
                "linear2.bias",
            ):
                src_key = f"{prefix}{suffix}"
                if src_key not in dense_sd:
                    raise KeyError(f"Missing Stage 1 FFN2 weight required for upcycling: {src_key}")
                src_tensor = dense_sd[src_key]

                comb_key = f"{prefix}combining_ffn.{suffix}"
                assert comb_key in moe_sd and moe_sd[comb_key].shape == src_tensor.shape, f"Shape mismatch on {comb_key}"
                moe_sd[comb_key].copy_(src_tensor)
                combining_cloned += 1
                populated_keys.add(comb_key)

                for exp_idx in range(num_experts):
                    exp_key = f"{prefix}experts.{exp_idx}.{suffix}"
                    assert exp_key in moe_sd and moe_sd[exp_key].shape == src_tensor.shape, f"Shape mismatch on {exp_key}"
                    moe_sd[exp_key].copy_(src_tensor)
                    experts_cloned += 1
                    populated_keys.add(exp_key)

    # Assert zero missing encoder FFN weights across all 12 layers
    missing_ffn = [
        k for k in moe_sd.keys()
        if ("ffn1" in k or "combining_ffn" in k or ".experts." in k or (".ffn2." in k and "router" not in k))
        and k not in populated_keys
    ]
    assert len(missing_ffn) == 0, f"Missing encoder FFN weights after upcycling: {missing_ffn}"

    moe_model.load_state_dict(moe_sd)
    msg = (
        f"Upcycled Stage 1 Backbone -> 130M Causal MoE: "
        f"{transferred} backbone tensors transferred, {combining_cloned} combining_ffn cloned, "
        f"{experts_cloned} expert tensors cloned (0 missing FFN weights)."
    )
    if log_path is not None:
        log_msg(log_path, msg)
    else:
        print(msg, flush=True)
    return moe_model


def configure_differential_optimizer(model: CausalStreamingASRModel, lr: float):
    """Freeze frontend & reconstruction_head; apply 0.1x LR to layers 1-4 and 1.0x LR to layers 5-12 + CTC."""
    for p in model.frontend.parameters():
        p.requires_grad = False
    if hasattr(model, "reconstruction_head") and model.reconstruction_head is not None:
        for p in model.reconstruction_head.parameters():
            p.requires_grad = False

    backbone_params: List[nn.Parameter] = []
    moe_params: List[nn.Parameter] = []

    for p in model.encoder.subsampling.parameters():
        backbone_params.append(p)
    for p in model.encoder.pos_enc.parameters():
        backbone_params.append(p)
    for layer_idx in range(12):
        target_list = backbone_params if layer_idx < 4 else moe_params
        for p in model.encoder.layers[layer_idx].parameters():
            target_list.append(p)
    for p in model.ctc_head.parameters():
        moe_params.append(p)

    trainable_params = backbone_params + moe_params
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": moe_params, "lr": lr},
        ],
        weight_decay=1e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    return optimizer, trainable_params, backbone_params, moe_params


@torch.no_grad()
def evaluate_heldout_cer(
    model: CausalStreamingASRModel,
    val_items: List[Dict[str, Any]],
    tokenizer: MarathiTokenizer,
    device: torch.device,
    batch_size: int = 16,
) -> float:
    """Fast held-out speaker-disjoint CER evaluation for early stopping and best_model selection."""
    from moe.respin_dataset import collate_moe_batch

    was_training = model.training
    model.eval()
    cers: List[float] = []
    for i in range(0, len(val_items), batch_size):
        batch = collate_moe_batch(val_items[i : i + batch_size], tokenizer, augment=False)
        if batch is None:
            continue
        audio = batch["audio"].to(device, non_blocking=True)
        audio_lens = batch["audio_lengths"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(audio, is_waveform=True, dialect_idx=None, use_hard_routing=False)["logits"]
        input_lengths = torch.clamp(audio_lens // (160 * 4), min=1, max=logits.size(1))
        for b in range(audio.size(0)):
            valid_t = int(input_lengths[b].item())
            pred_txt = tokenizer.ctc_decode(logits[b, :valid_t].argmax(dim=-1).tolist())
            cers.append(compute_cer(batch["texts"][b], pred_txt))
    if was_training:
        model.train()
    return sum(cers) / max(len(cers), 1)


def run_stage2_moe_adaptation(
    stage1_ckpt_path: Optional[str] = None,
    total_steps: int = 300000,
    batch_size: int = 32,
    grad_accum_steps: int = 4,
    lr: float = 1.5e-4,
    warmup_steps: int = 1000,
    ctc_warmup_steps: int = 3000,
    phase4a_steps: int = 15000,
    aux_weight: float = 0.01,
    save_every: int = 1000,
    log_every: int = 25,
    keep_last_k: int = 3,
    left_context: int = 128,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 0.001,
    resume_path: Optional[str] = None,
) -> None:
    dirs = setup_experiment_dir()
    train_log = dirs["logs"] / "train.log"

    s1_ckpt = stage1_ckpt_path or str(DEFAULT_STAGE1_CKPT)
    if not os.path.exists(s1_ckpt):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {s1_ckpt}")

    log_msg(train_log, "=" * 70)
    log_msg(train_log, "STARTING STAGE 2: CAUSAL MOE DIALECT ADAPTATION")
    log_msg(
        train_log,
        f"Max Steps: {total_steps} | Batch: {batch_size} x Accum {grad_accum_steps} | Peak LR: {lr} | "
        f"CTC Warmup: {ctc_warmup_steps} | Phase 4A (Hard): {phase4a_steps} | "
        f"Left Context: {left_context} | Early Stop Patience: {early_stop_patience} evals",
    )
    log_msg(train_log, "=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    log_msg(train_log, f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    run = sentinel.init(
        project="causal_conversion_130m",
        run_name="stage2_causal_moe_adapt",
        config={
            "stage": 2,
            "architecture": "CausalStreamingASRModel (130M MoE, strictly causal)",
            "left_context": left_context,
            "num_experts": 3,
            "moe_layers": "5-12",
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "total_steps": total_steps,
            "ctc_warmup_steps": ctc_warmup_steps,
            "phase4a_steps": phase4a_steps,
            "aux_weight": aux_weight,
            "lr": lr,
            "early_stop_patience": early_stop_patience,
            "base_checkpoint": s1_ckpt,
        },
    )

    # 1. Build Strictly Causal Model & Upcycle from Stage 1
    model = upcycle_stage1_to_moe(
        s1_ckpt,
        vocab_size=105,
        num_experts=3,
        moe_start_layer=5,
        left_context=left_context,
        log_path=train_log,
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
    optimizer, trainable_params, backbone_params, moe_params = configure_differential_optimizer(model, lr=lr)

    total_opt_steps = max(1, total_steps // grad_accum_steps)
    warmup_opt_steps = max(1, warmup_steps // grad_accum_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_opt_steps, total_steps=total_opt_steps, min_lr_ratio=0.05)

    log_msg(
        train_log,
        f"Trainable params: {sum(p.numel() for p in trainable_params):,} "
        f"(Backbone 0.1x LR: {sum(p.numel() for p in backbone_params):,}, "
        f"MoE+CTC 1.0x LR: {sum(p.numel() for p in moe_params):,})",
    )

    step = 0
    rolling_ctc_loss = 0.0
    rolling_cer = 1.0
    best_cer = float("inf")
    early_stop_counter = 0
    early_stop_min_steps = ctc_warmup_steps + phase4a_steps + 5000

    if resume_path and os.path.exists(resume_path):
        r_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(r_ckpt["model_state_dict"])
        if "optimizer_state_dict" in r_ckpt:
            optimizer.load_state_dict(r_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in r_ckpt:
            scheduler.load_state_dict(r_ckpt["scheduler_state_dict"])
        step = int(r_ckpt.get("step", 0))
        rolling_ctc_loss = float(r_ckpt.get("ctc_loss", 0.0))
        best_cer = float(r_ckpt.get("best_cer", r_ckpt.get("val_cer", r_ckpt.get("cer", float("inf")))))
        rolling_cer = float(r_ckpt.get("cer", 1.0))
        early_stop_counter = int(r_ckpt.get("early_stop_counter", 0))
        log_msg(
            train_log,
            f"Resumed Stage 2 from {resume_path} at step {step} "
            f"(best val CER: {best_cer:.4f}, patience: {early_stop_counter}/{early_stop_patience})",
        )

    if step >= total_steps:
        log_msg(train_log, f"Stage 2 already at step {step} >= {total_steps}. Skipping training.")
        run.finish()
        return

    # 2. Load Speaker-Disjoint RESPIN Training Split & Fixed Held-out Validation Subset
    tokenizer = MarathiTokenizer(str(VOCAB_PATH))
    train_utts, calib_utts = build_or_load_speaker_split(cache_path=str(RESPIN_CACHE_PATH))
    val_subset = calib_utts[:128]
    log_msg(
        train_log,
        f"RESPIN speaker-disjoint split loaded in fast path: "
        f"{len(train_utts):,} train utterances | {len(calib_utts):,} held-out calib ({len(val_subset)} used for early stopping).",
    )
    loader = LocalRESPINPrefetchLoader(
        train_utts,
        tokenizer,
        batch_size=batch_size,
        queue_size=6,
        shuffle=True,
        bucket_multiplier=4,
    )

    ctc_criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    spec_aug = SpecAugment(freq_mask_max=27, time_mask_max=40, num_freq_masks=2, num_time_masks=5).to(device)
    moe_layers = [4, 5, 6, 7, 8, 9, 10, 11]

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
            dialect_indices = batch["dialect_indices"].to(device, non_blocking=True)

            is_warmup = step <= ctc_warmup_steps
            is_phase_4a = (not is_warmup) and (step <= ctc_warmup_steps + phase4a_steps)

            if (step - 1) % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    with torch.no_grad():
                        spec = model.frontend(audio)
                        spec = spec_aug(spec)

                    outputs = model(
                        spec,
                        is_waveform=False,
                        dialect_idx=dialect_indices if is_phase_4a else None,
                        use_hard_routing=is_phase_4a,
                    )
                    logits = outputs["logits"]
                    log_probs = F.log_softmax(logits, dim=-1)
                    log_probs_t = log_probs.transpose(0, 1)  # (T, B, V)

                    input_lengths = torch.clamp(audio_lens // (160 * 4), min=1, max=logits.size(1))
                    ctc_loss = ctc_criterion(log_probs_t, targets, input_lengths, target_lens)

                    total_aux_loss = torch.tensor(0.0, device=device)
                    if not is_warmup:
                        for layer_idx in moe_layers:
                            ffn2 = model.encoder.layers[layer_idx].ffn2
                            if isinstance(ffn2, SparseMoELayer) and hasattr(ffn2, "current_aux_loss"):
                                total_aux_loss = total_aux_loss + ffn2.current_aux_loss

                    total_loss = ctc_loss + aux_weight * total_aux_loss

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

            ctc_val = float(ctc_loss.item())
            aux_val = float(total_aux_loss.item()) if isinstance(total_aux_loss, torch.Tensor) else 0.0
            rolling_ctc_loss = 0.9 * rolling_ctc_loss + 0.1 * ctc_val if rolling_ctc_loss > 0 else ctc_val

            if is_warmup:
                phase_name = "0 (CTC Warmup)"
            elif is_phase_4a:
                phase_name = "4A (Hard Routing)"
            else:
                phase_name = "4B (Dynamic MoE)"

            if step % log_every == 0 or step == 1 or step % save_every == 0 or step == total_steps:
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

            if step % log_every == 0 or step == 1:
                curr_lr = optimizer.param_groups[1]["lr"]
                log_msg(
                    train_log,
                    f"Step {step:5d}/{total_steps} | Phase: {phase_name} | "
                    f"CTC: {rolling_ctc_loss:.4f} | Aux: {aux_val:.4f} | CER: {last_cer:.3f} (Roll: {rolling_cer:.3f}) | "
                    f"GradNorm: {grad_norm_val:.2f} | Thru: {throughput:5.1f} aud-s/s | LR: {curr_lr:.2e}",
                )
                log_msg(train_log, f"  [Ref ]: {ref_text[:80]}")
                log_msg(train_log, f"  [Pred]: {pred_text[:80]}")

                run.log({
                    "total_loss": float(total_loss.item()),
                    "ctc_loss": rolling_ctc_loss,
                    "aux_loss": aux_val,
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
                    "ctc_loss": rolling_ctc_loss,
                    "aux_loss": aux_val,
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
                ckpt_path = dirs["checkpoints"] / f"causal_moe_stage2_step_{step}.pt"
                best_path = dirs["checkpoints"] / "best_model.pt"

                if improved or not best_path.exists():
                    best_cer = min(best_cer, val_cer)
                    early_stop_counter = 0
                    ckpt_payload["best_cer"] = best_cer
                    ckpt_payload["early_stop_counter"] = early_stop_counter
                    torch.save(ckpt_payload, ckpt_path)
                    torch.save(ckpt_payload, best_path)
                    log_msg(train_log, f"Updated best Stage 2 checkpoint (val CER={best_cer:.4f}) -> {best_path}")
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
                        log_msg(train_log, f"[Early Stop] Triggered at step {step}! Stopping Stage 2 cleanly.")
                        prune_step_checkpoints(dirs["checkpoints"], "causal_moe_stage2_step_*.pt", keep_last_k=keep_last_k)
                        break

                prune_step_checkpoints(dirs["checkpoints"], "causal_moe_stage2_step_*.pt", keep_last_k=keep_last_k)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        loader.close()
        run.finish()

    elapsed_hrs = (time.perf_counter() - start_time) / 3600.0
    log_msg(train_log, f"Stage 2 Complete in {elapsed_hrs:.2f} hrs! Best checkpoint: {dirs['checkpoints'] / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Causal MoE Dialect Adaptation")
    parser.add_argument("--stage1-ckpt", type=str, default=str(DEFAULT_STAGE1_CKPT))
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--ctc-warmup-steps", type=int, default=3000)
    parser.add_argument("--phase4a-steps", type=int, default=15000)
    parser.add_argument("--aux-weight", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-last-k", type=int, default=3)
    parser.add_argument("--left-context", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    run_stage2_moe_adaptation(
        stage1_ckpt_path=args.stage1_ckpt,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        ctc_warmup_steps=args.ctc_warmup_steps,
        phase4a_steps=args.phase4a_steps,
        aux_weight=args.aux_weight,
        save_every=args.save_every,
        log_every=args.log_every,
        keep_last_k=args.keep_last_k,
        left_context=args.left_context,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        resume_path=args.resume,
    )
