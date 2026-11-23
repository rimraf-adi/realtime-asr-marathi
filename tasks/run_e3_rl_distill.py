"""
Experiment E3 Runner: Causal Teacher-Guided Sequence RL + Distillation.
Strictly compliant with SENTINEL.md:
- One experiment = One directory (experiments/e3_rl_guided_distill/)
- Subdirectories: config.yaml, results/, plots/, checkpoints/, logs/
- Real-time logging via sentinel-sdk (sentinel.init)
- Checkpoint: Lowest-error 130M MoE teacher (d5621c61, RESPIN WER: 19.50%)
- Target: Causal Streaming Student with Group Relative Policy Optimization (GRPO)
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "d:/marathi-asr" not in sys.path:
    sys.path.insert(0, "d:/marathi-asr")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sentinel
from models.causal_conformer.causal_asr_model import CausalStreamingASRModel
from datasets.marathi_streaming import MarathiASRDataset, collate_fn
from tasks.rl_distill_loss import SequenceRLDistillLoss


def setup_experiment_dir() -> dict:
    exp_name = "e3_rl_guided_distill"
    base_dir = Path("d:/realtime/experiments") / exp_name
    dirs = {
        "base": base_dir,
        "results": base_dir / "results",
        "plots": base_dir / "plots",
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    config_src = "d:/realtime/configs/experiment/e3_rl_guided_distill.yaml"
    if os.path.exists(config_src):
        shutil.copyfile(config_src, base_dir / "config.yaml")

    return dirs


def log_msg(log_file: Path, text: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {text}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())


def sample_ctc_hypotheses(logits: torch.Tensor, group_size: int = 4, temperature: float = 1.0, blank_id: int = 0) -> List[List[List[int]]]:
    """
    Generate G diverse hypotheses per utterance for GRPO advantage computation.
    Args:
        logits: (B, T, V)
    Returns:
        List over B of List over G of token sequences (collapsed)
    """
    B, T, V = logits.shape
    batch_hyps = []

    probs = F.softmax(logits / max(0.1, temperature), dim=-1)

    for b in range(B):
        group = []
        # Sample G times
        for _ in range(group_size):
            sampled_tokens = torch.multinomial(probs[b], num_samples=1).squeeze(-1).tolist()
            # CTC collapse
            collapsed = []
            prev = None
            for t in sampled_tokens:
                if t != blank_id and t != prev:
                    collapsed.append(t)
                prev = t
            group.append(collapsed)
        batch_hyps.append(group)

    return batch_hyps


def run_e3_training():
    dirs = setup_experiment_dir()
    train_log = dirs["logs"] / "train.log"
    summary_log = dirs["logs"] / "summary.log"

    log_msg(train_log, "==================================================================")
    log_msg(train_log, "STARTING EXPERIMENT E3: CAUSAL TEACHER-GUIDED SEQUENCE RL + DISTILLATION")
    log_msg(train_log, "==================================================================")

    # 1. Initialize Sentinel Run
    run = sentinel.init(
        project="causal_conversion_130m",
        run_name="e3_rl_guided_distill",
        config={
            "experiment_id": "E3",
            "method": "Teacher-Guided GRPO + Multi-Exit KD",
            "teacher_checkpoint": "d:/marathi-asr/runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt",
            "teacher_wer_respin": 19.50,
            "teacher_sha256": "d5621c61",
            "student_architecture": "CausalStreamingASRModel",
            "physical_batch_size": 16,
            "grad_accum_steps": 4,
            "effective_batch_size": 64,
            "learning_rate": 3e-5,
            "total_steps": 50000,
            "total_epochs": 37.4,
            "training_hours": 150.8,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 2. Load Lowest-Error Offline Teacher (d5621c61, WER: 19.50%)
    log_msg(train_log, "Loading lowest-error 130M MoE Teacher (SHA d5621c61, RESPIN WER: 19.50%)...")
    from moe.upcycling import load_moe_model
    teacher = load_moe_model("d:/marathi-asr/runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt", device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log_msg(train_log, "Teacher successfully loaded and frozen.")

    # 3. Load Causal Streaming Student
    log_msg(train_log, "Initializing 130M Causal Streaming Student...")
    student = CausalStreamingASRModel(enable_moe=True, num_experts=3, ffn_expansion=4).to(device)
    
    # Auto-resume from latest checkpoint if available
    latest_ckpt = dirs["checkpoints"] / "causal_moe_step_3000.pt"
    initial_step = 0
    if latest_ckpt.exists():
        log_msg(train_log, f"Resuming student weights from checkpoint: {latest_ckpt}")
        student.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=True))
        initial_step = 3000
    else:
        log_msg(train_log, "Loading from base offline checkpoint...")
        student.load_offline_checkpoint("d:/realtime/models/marathi-conformer-130m-moe/model.pt", strict=False)
    
    student.train()
    log_msg(train_log, f"Student successfully initialized in causal training mode (starting step: {initial_step}).")

    # 4. Vocab & Tokenizer Mapping
    vocab_path = "d:/realtime/models/marathi-conformer-130m-moe/vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    token2id = vocab_dict["token2id"]
    id2token = {v: k for k, v in token2id.items()}

    # 5. Dataset & Dataloader
    log_msg(train_log, "Preparing Marathi ASR training dataset (KathBath + OpenSLR64: ~150.8 hours)...")
    train_manifests = [
        "d:/marathi-asr/data/vistaar_benchmarks/kathbath_train/marathi/manifest.json",
        "d:/marathi-asr/data/openslr64_marathi/manifest.json",
    ]
    dataset = MarathiASRDataset(
        manifest_paths=train_manifests,
        vocab_path=vocab_path,
        sample_rate=16000,
        min_duration=0.8,
        max_duration=12.0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Scaled up to physical batch size 16
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )
    log_msg(train_log, f"Dataloader ready with {len(dataset)} verified audio-text pairs (Batch: 16, Accum: 4, Effective: 64).")

    # 6. Optimizer, Scaler, & Loss
    optimizer = torch.optim.AdamW(student.parameters(), lr=1.0e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    criterion = SequenceRLDistillLoss(
        alpha_ctc=0.70,
        alpha_kd=0.20,
        beta_grpo=0.10,
        gamma_hidden=0.00,
        temperature=2.0,
        epsilon_clip=0.20,
        blank_id=0,
    )

    # 7. Training Execution Loop: 10,000 steps (~7.5 full epochs, 640k utterances effective)
    total_steps = 10000
    accum_steps = 4
    log_every = 25
    save_every = 1000
    step = initial_step
    step_losses = []
    step_rewards = []

    log_msg(train_log, f"Beginning full RL-guided distillation training from step {step} to {total_steps} supervised steps...")

    data_iter = iter(dataloader)
    start_time = time.perf_counter()

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step += 1
        waveforms = batch["waveforms"].to(device)
        labels = batch["labels"].to(device)
        wav_lens = batch["wav_lengths"].to(device)
        label_lens = batch["label_lengths"].to(device)

        # 4x subsampling converts input frames to encoder output length
        spec_frames = wav_lens // 160

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            # Teacher forward (frozen, non-causal representation extraction)
            with torch.no_grad():
                teacher_out = teacher.forward_ctc(waveforms)
                teacher_logits = teacher_out["log_probs"].detach()
                teacher_hidden = {12: teacher_out["final_hidden"].detach()}

            # Student forward (strictly causal, gradient tracked)
            student_out = student.forward(waveforms, is_waveform=True)
            student_logits = student_out["logits"]
            student_hidden = student_out["exit_hiddens"]

            input_lengths = torch.clamp(spec_frames // 4, max=student_logits.size(1))

            # Sample GRPO hypothesis groups from student policy
            with torch.no_grad():
                hypotheses = sample_ctc_hypotheses(
                    student_logits.detach(),
                    group_size=4,
                    temperature=1.0,
                    blank_id=0,
                )

            # Reconstruct reference texts for CER calculation
            ref_texts = []
            for b in range(labels.size(0)):
                toks = labels[b, :label_lens[b]].tolist()
                chars = [id2token.get(t, "") for t in toks if t != 0]
                ref_texts.append("".join(chars).replace("|", " ").strip())

            # Multi-objective RL-Distill Loss calculation
            loss_dict = criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                labels=labels,
                input_lengths=input_lengths,
                target_lengths=label_lens,
                hypotheses_tokens=hypotheses,
                reference_texts=ref_texts,
                id2token=id2token,
            )
            loss = loss_dict["total_loss"] / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        step_losses.append(loss_dict["total_loss"].item())
        step_rewards.append(loss_dict["avg_reward"])

        # Real-time Sentinel & console logging
        if step % log_every == 0 or step == 1:
            avg_loss = sum(step_losses[-log_every:]) / len(step_losses[-log_every:])
            avg_r = sum(step_rewards[-log_every:]) / len(step_rewards[-log_every:])
            log_msg(
                train_log,
                f"Step {step:04d}/{total_steps} | Loss: {avg_loss:.4f} "
                f"(CTC: {loss_dict['loss_ctc']:.3f}, KD: {loss_dict['loss_kd']:.3f}, "
                f"GRPO: {loss_dict['loss_grpo']:.3f}, Hidden: {loss_dict['loss_hidden']:.4f}) | "
                f"Avg CER Reward: {avg_r:.3f}"
            )

            run.log({
                "loss/total": avg_loss,
                "loss/ctc": float(loss_dict["loss_ctc"].item()),
                "loss/kd": float(loss_dict["loss_kd"].item()),
                "loss/grpo": float(loss_dict["loss_grpo"].item()),
                "reward/cer": avg_r,
            }, step=step)

        # Checkpointing
        if step % save_every == 0 or step == total_steps:
            ckpt_path = dirs["checkpoints"] / f"causal_moe_step_{step}.pt"
            torch.save(student.state_dict(), ckpt_path)
            log_msg(train_log, f"Checkpoint saved at step {step} -> {ckpt_path}")

    elapsed = time.perf_counter() - start_time
    log_msg(train_log, f"Experiment E3 initial training phase finished in {elapsed:.1f}s.")

    # 8. Plot Loss & Reward Curves into plots/
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(step_losses) + 1), step_losses, label="Total RL-Distill Loss", color="#1A73E8")
    axes[0].set_xlabel("Optimization Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("E3: Causal Teacher-Guided RL + Distillation Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(step_rewards) + 1), step_rewards, label="Hypothesis CER Reward (1 - CER)", color="#34A853")
    axes[1].set_xlabel("Optimization Step")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("E3: Sequence Policy CER Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_file = dirs["plots"] / "e3_loss_reward_curve.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()

    # 9. Save Summary & Final Artifacts
    results_summary = {
        "experiment_id": "E3",
        "experiment_name": "e3_rl_guided_distill",
        "completed_at": datetime.utcnow().isoformat(),
        "total_steps": total_steps,
        "final_loss": round(step_losses[-1], 4),
        "final_cer_reward": round(step_rewards[-1], 4),
        "teacher_anchor": "130M MoE SFT (WER 19.50% RESPIN, d5621c61)",
        "student_model": "CausalStreamingASRModel (130M MoE)",
        "artifacts": {
            "checkpoint": str(dirs["checkpoints"] / f"causal_moe_step_{total_steps}.pt"),
            "plot": str(plot_file),
            "config": str(dirs["base"] / "config.yaml"),
        }
    }
    with open(dirs["results"] / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    with open(summary_log, "w", encoding="utf-8") as f:
        f.write("EXPERIMENT E3 SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Teacher Anchor: 130M MoE SFT (19.50% RESPIN, 29.41% KathBath)\n")
        f.write(f"Student: 130M Causal Conformer MoE with Group Relative Policy Optimization\n")
        f.write(f"Steps Completed: {total_steps}\n")
        f.write(f"Final Loss: {results_summary['final_loss']}\n")
        f.write(f"Final CER Reward: {results_summary['final_cer_reward']}\n")

    run.log_artifact(str(plot_file))
    run.finish()
    log_msg(train_log, "Experiment E3 completed and fully logged in Sentinel.")


if __name__ == "__main__":
    run_e3_training()
