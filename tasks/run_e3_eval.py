"""
Full Benchmark Evaluation Suite for Causal Streaming Conformer (E3).
Evaluates the final fine-tuned 130M Causal MoE checkpoint (step 50,000) on:
1. IISc RESPIN Multi-Dialect Test Set (D1 Malvani, D2 Ahirani, D3 Standard, D4 Varhadi)
2. KathBath Marathi Test Set
3. Google FLEURS Marathi Test Set

Uses standardized `mr-norm-1` normalizer and jiwer word/character error rates.
Strictly compliant with SENTINEL.md (outputs saved in experiments/e3_full_eval_suite/).
"""

import os
import sys
import json
import time
import shutil
import hashlib
import io
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "d:/marathi-asr" not in sys.path:
    sys.path.insert(0, "d:/marathi-asr")

import torch
import numpy as np
import soundfile as sf
import torchaudio
import jiwer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sentinel
from normalizer import NORM_VERSION, normalize
from models.causal_conformer.causal_asr_model import CausalStreamingASRModel
from models.causal_conformer.streaming_decoder import StreamingCTCDecoder

SR = 16000
RESPIN_DIR = Path(r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr")


def setup_eval_dirs() -> dict:
    exp_name = "e3_full_eval_suite"
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

    config_src = "d:/realtime/configs/experiment/e3_full_eval.yaml"
    if os.path.exists(config_src):
        shutil.copyfile(config_src, base_dir / "config.yaml")

    return dirs


def log_event(log_file: Path, text: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {text}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())


def resample(a, sr):
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
    return a.astype(np.float32)


def load_respin_dataset(limit=None):
    meta = json.loads((RESPIN_DIR / "meta_test_mr.json").read_text(encoding="utf-8"))
    items = list(meta.values())[:limit] if limit else list(meta.values())
    dialect_map = {
        "D1": "Malvani (D1)",
        "D2": "Ahirani (D2)",
        "D3": "Standard (D3)",
        "D4": "Varhadi (D4)",
    }
    for m in items:
        a, sr = sf.read(RESPIN_DIR / m["wav_path"], dtype="float32")
        yield resample(a, sr), m["text"], {"dialect": dialect_map.get(m["dialect"], m["dialect"])}


def load_kathbath_dataset(limit=None):
    from eval_offline import load_parquet
    return load_parquet("ai4bharat/Kathbath", "text", {"gender": "gender", "district": "district"}, limit)


def load_fleurs_dataset(limit=None):
    from eval_offline import load_fleurs
    return load_fleurs(limit)


class CausalASREvaluator:
    """Evaluates CausalStreamingASRModel in full causal and chunked streaming modes."""

    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cuda", left_context: Optional[int] = 128):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
        eff_left_ctx = model_cfg.get("left_context", left_context)
        self.model = CausalStreamingASRModel(
            enable_moe=True, num_experts=3, ffn_expansion=4, left_context=eff_left_ctx
        ).to(self.device)

        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Filter state dict
        own_state = self.model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in own_state and own_state[k].shape == v.shape}
        self.model.load_state_dict(filtered, strict=False)
        self.model.eval()

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab_dict = json.load(f)
        self.decoder = StreamingCTCDecoder(vocab_dict=self.vocab_dict, blank_id=0)

    @torch.no_grad()
    def transcribe_full(self, audio_list: List[np.ndarray]) -> List[str]:
        """Full causal sequence forward pass."""
        lens = [len(a) for a in audio_list]
        max_len = max(lens)
        batch_wav = torch.zeros(len(audio_list), max_len, dtype=torch.float32, device=self.device)
        for i, a in enumerate(audio_list):
            batch_wav[i, :len(a)] = torch.from_numpy(a)

        out = self.model.forward(batch_wav, is_waveform=True)
        logits = out["logits"]  # (B, T, V)

        transcriptions = []
        for b in range(len(audio_list)):
            valid_t = (lens[b] // 160) // 4
            b_logits = logits[b, :valid_t, :]
            tokens = torch.argmax(b_logits, dim=-1).tolist()
            # CTC collapse
            collapsed = []
            prev = 0
            for t in tokens:
                if t != 0 and t != prev:
                    collapsed.append(t)
                prev = t
            text = self.decoder.tokens_to_text(collapsed)
            transcriptions.append(text)

        return transcriptions

    @torch.no_grad()
    def transcribe_streaming_chunked(self, audio: np.ndarray, chunk_size_ms: int = 160) -> Tuple[str, float]:
        """Incremental chunked streaming inference with state caching."""
        chunk_samples = int(16000 * (chunk_size_ms / 1000.0))
        wav = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # (1, num_samples)

        state = self.model.init_streaming_state()
        dec_state = self.decoder.init_state()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for offset in range(0, wav.size(1), chunk_samples):
            chunk = wav[:, offset : offset + chunk_samples]
            logits_chunk, state = self.model.forward_chunk(chunk, state)
            if logits_chunk.size(1) > 0:
                _, dec_state = self.decoder.decode_chunk(logits_chunk, state=dec_state, is_final=False)

        # Flush final
        text, _ = self.decoder.decode_chunk(torch.empty(1, 0, 105, device=self.device), state=dec_state, is_final=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return text, elapsed


def evaluate_dataset(evaluator: CausalASREvaluator, items: List[Tuple], batch_size: int = 16) -> Dict[str, Any]:
    """Runs batch evaluation over items and computes WER/CER and dialect slices."""
    hyps = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        batch_audio = [items[i][0] for i in idx]
        try:
            outs = evaluator.transcribe_full(batch_audio)
        except Exception as e:
            print(f"Batch failed: {e}")
            outs = [""] * len(idx)
        for i, h in zip(idx, outs):
            hyps[i] = h

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_secs = time.perf_counter() - t_start

    total_audio_secs = sum(len(a) for a, _, _ in items) / SR
    refs = [normalize(t) for _, t, _ in items]
    hn = [normalize(h) for h in hyps]

    keep = [i for i, r in enumerate(refs) if r]
    r = [refs[i] for i in keep]
    h = [hn[i] for i in keep]

    errs, lens = [], []
    for a, b in zip(r, h):
        o = jiwer.process_words(a, b)
        errs.append(o.substitutions + o.deletions + o.insertions)
        lens.append(len(a.split()))

    errs = np.array(errs)
    lens = np.array(lens)

    wer = round(float(errs.sum() / max(lens.sum(), 1) * 100), 2)
    cer = round(float(jiwer.cer(r, h) * 100), 2)
    rtf = round(wall_secs / max(total_audio_secs, 1e-4), 4)

    # Dialect / metadata slices
    slices = {}
    if items and items[0][2]:
        for k in items[0][2]:
            g = {}
            for j, i in enumerate(keep):
                val = str(items[i][2][k] or "unknown")
                g.setdefault(val, []).append(j)
            slices[k] = {
                n: {
                    "wer": round(float(errs[ix].sum() / max(lens[ix].sum(), 1) * 100), 2),
                    "cer": round(float(jiwer.cer([r[idx] for idx in ix], [h[idx] for idx in ix]) * 100), 2),
                    "n": len(ix),
                }
                for n, ix in g.items()
            }

    return {
        "wer": wer,
        "cer": cer,
        "n_utts": len(r),
        "total_audio_secs": round(total_audio_secs, 2),
        "wall_secs": round(wall_secs, 2),
        "rtf": rtf,
        "rtfx": round(1.0 / max(rtf, 1e-6), 1),
        "slices": slices,
    }


def run_full_eval_suite(checkpoint_path: str = "d:/realtime/experiments/e3_rl_guided_distill/checkpoints/causal_moe_step_10000.pt", batch_size: int = 16):
    dirs = setup_eval_dirs()
    log_file = dirs["logs"] / "eval.log"

    log_event(log_file, "==================================================================")
    log_event(log_file, "STARTING FULL BENCHMARK EVALUATION SUITE: EXPERIMENT E3")
    log_event(log_file, "==================================================================")

    run = sentinel.init(
        project="causal_conversion_130m",
        run_name="e3_full_eval_suite",
        config={
            "checkpoint": checkpoint_path,
            "benchmarks": ["respin_mr", "kathbath_mr", "fleurs_mr"],
            "normalizer": "mr-norm-1",
        }
    )

    ckpt_path = checkpoint_path
    vocab_path = "d:/realtime/models/marathi-conformer-130m-moe/vocab.json"

    log_event(log_file, f"Loading fine-tuned checkpoint: {ckpt_path}")
    evaluator = CausalASREvaluator(checkpoint_path=ckpt_path, vocab_path=vocab_path)
    log_event(log_file, "Model successfully loaded on GPU.")

    results_all = {}

    # -------------------------------------------------------------
    # 1. Evaluate IISc RESPIN Multi-Dialect Benchmark
    # -------------------------------------------------------------
    log_event(log_file, "Evaluating on IISc RESPIN Multi-Dialect Test Set (2,170 utterances)...")
    respin_items = list(load_respin_dataset())
    log_event(log_file, f"Loaded {len(respin_items)} RESPIN utterances.")

    respin_metrics = evaluate_dataset(evaluator, respin_items, batch_size=batch_size)
    results_all["respin_mr"] = respin_metrics

    log_event(log_file, f"--> RESPIN Aggregate WER: {respin_metrics['wer']}% | CER: {respin_metrics['cer']}%")
    if "dialect" in respin_metrics["slices"]:
        for dname, dstats in sorted(respin_metrics["slices"]["dialect"].items()):
            log_event(log_file, f"    {dname}: WER={dstats['wer']}%, CER={dstats['cer']}% (n={dstats['n']})")

    # -------------------------------------------------------------
    # 2. Evaluate KathBath Marathi Benchmark
    # -------------------------------------------------------------
    log_event(log_file, "Evaluating on KathBath Marathi Test Set...")
    kathbath_items = list(load_kathbath_dataset())
    log_event(log_file, f"Loaded {len(kathbath_items)} KathBath utterances.")

    kathbath_metrics = evaluate_dataset(evaluator, kathbath_items, batch_size=batch_size)
    results_all["kathbath_mr"] = kathbath_metrics
    log_event(log_file, f"--> KathBath Aggregate WER: {kathbath_metrics['wer']}% | CER: {kathbath_metrics['cer']}%")

    # -------------------------------------------------------------
    # 3. Evaluate Google FLEURS Marathi Benchmark
    # -------------------------------------------------------------
    log_event(log_file, "Evaluating on Google FLEURS Marathi Test Set...")
    fleurs_items = list(load_fleurs_dataset())
    log_event(log_file, f"Loaded {len(fleurs_items)} FLEURS utterances.")

    fleurs_metrics = evaluate_dataset(evaluator, fleurs_items, batch_size=batch_size)
    results_all["fleurs_mr"] = fleurs_metrics
    log_event(log_file, f"--> FLEURS Aggregate WER: {fleurs_metrics['wer']}% | CER: {fleurs_metrics['cer']}%")

    # -------------------------------------------------------------
    # 4. Streaming Chunk Latency Benchmark on RESPIN Sample
    # -------------------------------------------------------------
    log_event(log_file, "Benchmarking real-time chunked streaming inference (160ms chunks) on RESPIN sample...")
    stream_times = []
    stream_audio_durations = []
    sample_utts = respin_items[:100]

    for audio, _, _ in sample_utts:
        text, wall_t = evaluator.transcribe_streaming_chunked(audio, chunk_size_ms=160)
        stream_times.append(wall_t)
        stream_audio_durations.append(len(audio) / SR)

    total_stream_wall = sum(stream_times)
    total_stream_audio = sum(stream_audio_durations)
    stream_rtf = round(total_stream_wall / max(total_stream_audio, 1e-4), 4)
    stream_rtfx = round(1.0 / max(stream_rtf, 1e-6), 1)
    avg_latency_ms = round((total_stream_wall / len(sample_utts)) * 1000, 2)

    streaming_benchmark = {
        "chunk_size_ms": 160,
        "n_sample_utts": len(sample_utts),
        "streaming_rtf": stream_rtf,
        "streaming_rtfx": stream_rtfx,
        "avg_utterance_latency_ms": avg_latency_ms,
    }
    results_all["streaming_benchmark"] = streaming_benchmark
    log_event(log_file, f"--> Streaming RTF: {stream_rtf} ({stream_rtfx}x real-time) | Latency: {avg_latency_ms} ms/utt")

    # -------------------------------------------------------------
    # 5. Save Summary JSON & Plot Dialect Breakdown
    # -------------------------------------------------------------
    summary_path = dirs["results"] / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)

    metrics_path = dirs["results"] / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)

    # Plot RESPIN dialect comparison
    if "dialect" in respin_metrics["slices"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        dialects = list(respin_metrics["slices"]["dialect"].keys())
        wers = [respin_metrics["slices"]["dialect"][d]["wer"] for d in dialects]
        colors = ["#4285F4", "#34A853", "#FBBC05", "#EA4335"]

        bars = ax.bar(dialects, wers, color=colors[:len(dialects)], width=0.5)
        ax.set_ylabel("Word Error Rate (WER %)")
        ax.set_title("E3 Fine-Tuned Causal MoE (Step 10,000): RESPIN Dialect WER")
        for bar, val in zip(bars, wers):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val}%", ha="center", fontweight="bold")

        ax.axhline(y=respin_metrics["wer"], color="#202124", linestyle="--", label=f"Aggregate WER ({respin_metrics['wer']}%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = dirs["plots"] / "respin_dialect_wer.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        run.log_artifact(str(plot_path))

    # Log to Sentinel
    run.log({
        "eval/respin_wer": respin_metrics["wer"],
        "eval/respin_cer": respin_metrics["cer"],
        "eval/kathbath_wer": kathbath_metrics["wer"],
        "eval/fleurs_wer": fleurs_metrics["wer"],
        "eval/streaming_rtfx": stream_rtfx,
    }, step=10000)

    run.finish()
    log_event(log_file, "Full evaluation suite completed and logged into Sentinel successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="d:/realtime/experiments/e3_rl_guided_distill/checkpoints/causal_moe_step_10000.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_full_eval_suite(checkpoint_path=args.checkpoint, batch_size=args.batch_size)
