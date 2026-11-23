"""
Marathi ASR Streaming Dataset for Causal Fine-tuning & Distillation.
Loads from NeMo-style manifest.json (audio_filepath, text, duration).
Supports speed perturbation, SpecAugment, and dynamic batching by duration.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


class MarathiASRDataset(Dataset):
    """
    Loads audio-text pairs from one or more manifest files.
    Returns raw waveforms and tokenized labels for CTC training.
    """

    def __init__(
        self,
        manifest_paths: List[str],
        vocab_path: str,
        sample_rate: int = 16000,
        min_duration: float = 0.5,
        max_duration: float = 20.0,
        speed_perturb: Optional[List[float]] = None,
    ):
        self.sample_rate = sample_rate
        self.speed_perturb = speed_perturb or [1.0]

        # Load vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.token2id = vocab["token2id"]
        self.blank_id = vocab.get("blank_id", 0)

        # Load all manifest entries
        self.entries: List[Dict] = []
        for mpath in manifest_paths:
            manifest_p = Path(mpath).resolve()
            # If path ends in kathbath_train/marathi/manifest.json, base is vistaar_benchmarks
            if "kathbath_train" in str(manifest_p):
                base_dir = manifest_p.parent.parent.parent
            else:
                base_dir = manifest_p.parent

            with open(mpath, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    dur = entry.get("duration", 0)
                    if dur < min_duration or dur > max_duration:
                        continue
                    # Resolve relative audio paths
                    audio_path = entry["audio_filepath"]
                    if not Path(audio_path).is_absolute():
                        resolved_path = str(base_dir / audio_path)
                        # Fallback check
                        if not Path(resolved_path).exists():
                            resolved_path = str(manifest_p.parent / audio_path)
                        audio_path = resolved_path

                    if Path(audio_path).exists():
                        self.entries.append({
                            "audio_filepath": audio_path,
                            "text": entry["text"],
                            "duration": dur,
                        })

        # Sort by duration for efficient batching
        self.entries.sort(key=lambda e: e["duration"])

    def __len__(self) -> int:
        return len(self.entries)

    def tokenize(self, text: str) -> List[int]:
        """Character-level tokenization matching vocab.json."""
        tokens = []
        i = 0
        while i < len(text):
            # Try 2-char tokens first (e.g., "क्ष", "ज्ञ")
            if i + 1 < len(text) and text[i:i+2] in self.token2id:
                tokens.append(self.token2id[text[i:i+2]])
                i += 2
            elif text[i] in self.token2id:
                tokens.append(self.token2id[text[i]])
                i += 1
            else:
                tokens.append(self.token2id.get("<unk>", 1))
                i += 1
        return tokens

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]

        # Load audio
        wav, sr = torchaudio.load(entry["audio_filepath"])
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav[0]  # mono

        # Speed perturbation (random during training)
        speed = random.choice(self.speed_perturb)
        if speed != 1.0:
            effects = [["speed", str(speed)], ["rate", str(self.sample_rate)]]
            wav_aug, _ = torchaudio.sox_effects.apply_effects_tensor(
                wav.unsqueeze(0), self.sample_rate, effects
            )
            wav = wav_aug.squeeze(0)

        # Tokenize text
        labels = self.tokenize(entry["text"])

        return {
            "waveform": wav,
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": entry["text"],
            "duration": entry["duration"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Dynamic-length collation with padding."""
    wav_lens = [item["waveform"].size(0) for item in batch]
    label_lens = [item["labels"].size(0) for item in batch]

    max_wav = max(wav_lens)
    max_label = max(label_lens)

    waveforms = torch.zeros(len(batch), max_wav)
    labels = torch.zeros(len(batch), max_label, dtype=torch.long)
    wav_lengths = torch.tensor(wav_lens, dtype=torch.long)
    label_lengths = torch.tensor(label_lens, dtype=torch.long)

    for i, item in enumerate(batch):
        waveforms[i, :wav_lens[i]] = item["waveform"]
        labels[i, :label_lens[i]] = item["labels"]

    return {
        "waveforms": waveforms,
        "labels": labels,
        "wav_lengths": wav_lengths,
        "label_lengths": label_lengths,
    }


class SpecAugment(torch.nn.Module):
    """SpecAugment data augmentation applied to log-mel spectrograms."""

    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 27,
        time_masks: int = 10,
        time_width_ratio: float = 0.05,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width_ratio = time_width_ratio

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (batch, time, freq)
        """
        if not self.training:
            return spec

        b, t, f = spec.shape
        max_time_width = max(1, int(t * self.time_width_ratio))

        for _ in range(self.freq_masks):
            fw = random.randint(0, min(self.freq_width, f - 1))
            f0 = random.randint(0, f - fw)
            spec[:, :, f0:f0 + fw] = 0.0

        for _ in range(self.time_masks):
            tw = random.randint(0, max_time_width)
            t0 = random.randint(0, max(0, t - tw))
            spec[:, t0:t0 + tw, :] = 0.0

        return spec
