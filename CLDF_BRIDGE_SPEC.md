# Closed-Loop Decoder-Feedback Bridge (CLDF-Bridge): Technical Specification

**Document Standard:** ASD-STE100 (Simplified Technical English)  
**Project:** Real-Time Streaming Multilingual ASR with Seq2Seq Decoders  
**Target Output:** Q1 Journal Manuscript (IEEE/ACM TASLP, Computer Speech & Language, Speech Communication, or ACM TALLIP)

---

## 1. Purpose and Scope

This document defines the **Closed-Loop Decoder-Feedback Bridge (CLDF-Bridge)**. The CLDF-Bridge connects a causal speech encoder to a modified chunk-causal `mT5` or `ByT5` text decoder.

Unlike all standard speech-to-text bridges, the CLDF-Bridge breaks the one-way projection bottleneck. The decoder sends its running linguistic state back into the bridge. The bridge uses this state to re-project incoming acoustic chunks for the active language, dialect, or script.

The design achieves four primary goals:
1. Build a real-time streaming Automatic Speech Recognition (ASR) system with tunable chunk latency (160 ms to 640 ms).
2. Handle multilingual shifts, dialect continuums, and mid-sentence code-switching with zero external Language-ID tags.
3. Exploit Seq2Seq cross-attention in a way that prefix-KV decoder-only Speech-LLMs cannot replicate.
4. Preserve zero future acoustic lookahead and low per-chunk compute overhead.

---

## 2. Technical Names and Definitions

Use these exact terms across the codebase and manuscript.

| Term / Symbol | Definition |
| :--- | :--- |
| **Open-Loop Bridge** | A standard one-way projector (`Encoder -> Bridge -> Decoder`) that maps audio frames without decoder context. |
| **Closed-Loop Bridge (CLDF-Bridge)** | A bidirectional bridge where past decoder hidden states modulate the acoustic projection of incoming audio chunks. |
| **Prefix-KV Lock-In** | The structural constraint of decoder-only Speech-LLMs where projected audio tokens stay frozen inside the prefix KV cache. |
| **$H^{(0)}$** | The base acoustic projection computed once from compressed speech encoder states. |
| **$u^{(c-1, l)}$** | The causal linguistic summary extracted from decoder layer $l$ across tokens emitted before chunk $c$. |
| **$H^{(c, l)}$** | The feedback-modulated acoustic representation fed into decoder cross-attention layer $l$ at chunk $c$. |
| **$\alpha_l$** | A learnable scalar feedback gate initialized to $0.0$ for layer $l$. |
| **`<WAIT>` Token** | A control token that stops decoder emission for chunk $c$ until audio chunk $c+1$ arrives. |

---

## 3. Problem Statement

### 3.1 The One-Way Bottleneck in Existing Bridges
Every major 2024–2026 speech-to-text bridge (`SLAM-ASR`, `MEUSLI`, `Q-Former`, `LegoSLM`, `SMEAR-MoE`, `SpeechLLM-XL`, `CSSEL-P2P`) uses a strictly open-loop pipeline:

$$\text{Speech Encoder} \xrightarrow{\text{one-way}} \text{Bridge} \xrightarrow{\text{one-way}} \text{Decoder}$$

An open-loop bridge must project raw acoustic frames into the decoder embedding space blindly. It receives zero feedback about what language, regional dialect, or script the decoder is currently writing.

This blind projection fails under three real-world rural multilingual shifts:
1. **Dialect Continuums:** Rural speakers span continuous phonological boundaries (for example, Varhadi between Marathi and Hindi, or Bhojpuri, Magahi, Maithili, and Chhattisgarhi near Hindi). Open-loop bridges default toward high-resource standard languages and erase dialect morphology.
2. **Intra-Utterance Code-Switching:** Rural Indian speakers frequently insert English or Hindi loanwords (*"KYC"*, *"subsidy"*, *"PM-KISAN"*, *"ration card"*) inside regional sentences. A static bridge cannot shift its acoustic focus mid-utterance.
3. **Cross-Script Ambiguity:** Shared Indic phonemes map to different scripts (Devanagari, Bengali, Kannada, Telugu). Without feedback from already-emitted script tokens, an open-loop bridge produces script mixing.

### 3.2 Why Decoder-Only Speech-LLMs Cannot Easily Copy This
In decoder-only streaming Speech-LLMs (`SpeechLLM-XL`, `CSSEL-P2P`), projected audio frames enter the self-attention prefix KV cache once. Re-projecting past or current audio features using deep decoder states requires recomputing the entire self-attention prefix.

In contrast, a Seq2Seq encoder–decoder (`mT5` or `ByT5`) reads acoustic features through **dedicated cross-attention layers** at every decoder depth $l$. The decoder can therefore modulate the acoustic bridge directly before cross-attention with minimal compute overhead.

### 3.3 Comparison Against Existing 2024–2026 Bridges

| Bridge Method | Information Flow | Handles Mid-Utterance Shift Without LID? | Streaming Seq2Seq Compatible? |
| :--- | :--- | :---: | :---: |
| **SLAM-ASR / MEUSLI** (2024–2026) | Open-loop (1-way) | No | Partial (heavy prefix) |
| **Q-Former** (2024–2025) | Open-loop (1-way) | No | Poor (non-monotonic) |
| **LegoSLM** (2025) | Open-loop (1-way CTC) | Limited | Partial |
| **SMEAR-MoE** (Jan 2026) | Open-loop (acoustic router only) | Acoustic-only | Partial |
| **SpeechLLM-XL / CSSEL-P2P** (2024–2026) | Open-loop chunked prefix | Acoustic-only | Decoder-only LLM |
| **CLDF-Bridge (Ours)** | **Closed-loop (Decoder -> Bridge)** | **Yes (acoustic + linguistic steering)** | **Native chunk-causal Seq2Seq** |

---

## 4. End-to-End Closed-Loop Streaming Architecture

The system processes incoming audio in fixed streaming chunks $c \in \{1, 2, \dots, C\}$ (160 ms, 320 ms, or 640 ms).

```mermaid
flowchart LR
    A["Audio Chunk c (160ms-320ms)"] --> B["Causal Speech Encoder (Cached States)"]
    B --> C["Causal Temporal Compressor"]
    C --> D["Base Bridge FFN_base -> H(c, 0)"]
    D --> E["Feedback Modulation Layer FFN_fb"]
    U["Past Decoder Linguistic State u(c-1, l)"] --> E
    E --> F["RMSNorm + Residual Gate tanh(alpha_l) -> H(c, l)"]
    F --> G["Append to Layer l Cross-Attention Cache"]
    G --> H["Modified Chunk-Causal mT5 / ByT5 Decoder (Layer l)"]
    H --> I["Emit Tokens for Chunk c until WAIT/EOS"]
    H -.->|"Update Causal State Cache for Chunk c+1"| U
```

### Streaming Lifecycle per Chunk $c$
1. The **Causal Speech Encoder** processes incoming audio chunk $c$ using cached attention keys, values, and convolution states.
2. The **Causal Temporal Compressor** downsamples chunk frames by stride $s \in \{2, 4\}$.
3. The **Base Bridge ($\text{FFN}_{\text{base}}$)** computes the open-loop base projection $H^{(c, 0)}$.
4. At each feedback-enabled decoder layer $l$, the **Feedback Modulation Layer ($\text{FFN}_{\text{fb}}$)** reads the cached linguistic summary $u^{(c-1, l)}$ from chunks $1..c-1$ and produces the steered projection $H^{(c, l)}$.
5. For chunk $c = 1$ (cold start), $u^{(0, l)}$ equals a learned initial vector $u_{\text{init}}^{(l)}$, so chunk 1 runs cleanly as an open-loop pass.
6. The **Modified `mT5`/`ByT5` Decoder** emits tokens for chunk $c$ until it predicts `<WAIT>` or `<EOS>`, and updates $u^{(c, l)}$ for chunk $c+1$.

---

## 5. Mathematical Specification of the CLDF-Bridge

Let $X^{(c)} \in \mathbb{R}^{B \times T_c \times d_{\text{enc}}}$ denote the speech encoder output for chunk $c$.

### 5.1 Step 1: Causal Temporal Compression and Base Projection
Apply a strictly causal 1D strided convolution (left-only padding) with stride $s \in \{2, 4\}$:

$$V^{(c)} = \text{SiLU}\!\big(\text{CausalConv1d}_s(X^{(c)})\big) \in \mathbb{R}^{B \times T'_c \times d_{\text{enc}}}$$

Compute the base acoustic projection once per chunk:

$$H^{(c, 0)} = W_{b2}\,\text{GELU}\!\big(W_{b1}\,\text{LN}(V^{(c)})\big) \in \mathbb{R}^{B \times T'_c \times d_{\text{dec}}}$$

### 5.2 Step 2: Causal Decoder Linguistic State Extraction
Let $S^{(c, l)} \in \mathbb{R}^{B \times N_c \times d_{\text{dec}}}$ denote the self-attention hidden states of decoder layer $l$ for the $N_c$ tokens emitted during chunk $c$.

Compute the mean linguistic state of chunk $c$:

$$\bar{s}^{(c, l)} = \frac{1}{\max(1, N_c)} \sum_{n=1}^{N_c} S^{(c, l)}_n$$

Update the causal running linguistic summary with an exponential moving average (EMA) with decay $\beta \in [0.5, 0.9]$:

$$u^{(c, l)} = \beta \cdot u^{(c-1, l)} + (1 - \beta) \cdot \bar{s}^{(c, l)}$$

If the decoder emits zero content tokens during chunk $c$ (silence chunk), keep $u^{(c, l)} = u^{(c-1, l)}$.

### 5.3 Step 3: Closed-Loop Feedback Modulation
At each feedback-enabled decoder layer $l \in \mathcal{L}_{\text{fb}}$ (for example, layers $\{2, 4, 6\}$ in a 6-layer `mT5-small` decoder), modulate the base bridge output using $u^{(c-1, l)}$:

$$H^{(c, l)} = \text{RMSNorm}\!\Big(H^{(c, 0)} + \tanh(\alpha_l) \cdot \text{FFN}_{\text{fb}}^{(l)}\big(H^{(c, 0)},\, u^{(c-1, l)}\big)\Big)$$

Define $\text{FFN}_{\text{fb}}^{(l)}$ as a fast, parameter-efficient **Gated FiLM-Bottleneck FFN**:

$$[\gamma_l,\, \delta_l] = W_{\text{cond}}^{(l)}\,\text{LN}(u^{(c-1, l)}) \in \mathbb{R}^{B \times 2 r_{\text{fb}}}$$

$$M^{(c, l)} = \big(1 + \gamma_l\big) \odot \text{GELU}\!\big(W_{\text{down}}^{(l)}\,H^{(c, 0)}\big) + \delta_l$$

$$\text{FFN}_{\text{fb}}^{(l)}\big(H^{(c, 0)},\, u^{(c-1, l)}\big) = W_{\text{up}}^{(l)}\,M^{(c, l)}$$

where $r_{\text{fb}} \ll d_{\text{dec}}$ (for example, $r_{\text{fb}} = 128$), and $\alpha_l$ starts at **$0.0$**.

**Stability Guarantee:** At training step 0, $\tanh(\alpha_l) = 0$. Therefore $H^{(c, l)} = \text{RMSNorm}(H^{(c, 0)})$. The pretrained `mT5`/`ByT5` cross-attention receives stable open-loop features immediately and smoothly activates closed-loop feedback as gradients flow.

### 5.4 Strict Causality Proof
Let $c$ be any audio chunk index:
- $H^{(c, 0)}$ depends only on audio chunks $1..c$ (causal encoder + causal convolution).
- $u^{(c-1, l)}$ depends only on decoder tokens emitted during chunks $1..c-1$.
- Decoder tokens emitted during chunk $c$ cross-attend only to cached bridge states $H^{(1..c, l)}$.
- Therefore, no token emitted at chunk $c$ accesses any audio frame from chunks $> c$. The entire closed-loop architecture is **strictly causal**.

### 5.5 Compute Overhead and Parallel Training
- **Streaming Inference Overhead:** At chunk $c$, $\text{FFN}_{\text{fb}}^{(l)}$ executes **only** on the new chunk's $T'_c$ frames (typically 2 to 8 frames for a 160 ms–320 ms chunk). Past chunk states $H^{(1..c-1, l)}$ stay cached inside each cross-attention layer. Total added latency is **less than 3%**.
- **Parallel GPU Training (No Slow Python Loop):** Under teacher forcing, all chunk boundaries and chunk token spans are known ahead of time from forced CTC alignment:
  1. Run a **fast Pass-0** (open-loop, $\alpha_l = 0$ or cached shifted embeddings) to extract teacher-forced chunk summaries $u^{(c-1, l)}$ for all chunks $c = 1..C$ in parallel!
  2. Broadcast $u^{(c-1, l)}$ to the acoustic frame indices of chunk $c$ in a single vectorized tensor operation.
  3. Run the **Closed-Loop Pass** in full GPU parallelism across all chunks and tokens!

---

## 6. Modified Chunk-Causal `mT5` / `ByT5` Decoder

### 6.1 Indic Vocabulary Pruning (`mT5-small`) vs. `ByT5-small`
- **Indic-Pruned `mT5-small`:** Filter the 250,112-token vocabulary down to ~20,000 Indic + Latin tokens (Devanagari, Bengali, Kannada, Telugu, English loanwords, digits, `<WAIT>`). This shrinks `mT5-small` from **300M to ~65M parameters**.
- **`ByT5-small` Track:** Keep the native 384-token UTF-8 byte vocabulary (plus `<WAIT>`). Byte-level decoding avoids subword shattering during rapid Marathi–English or Hindi–Bhojpuri code-switching.

### 6.2 Cross-Attention Layer Modification
In each feedback layer $l \in \mathcal{L}_{\text{fb}}$ of the `mT5`/`ByT5` decoder:
1. Replace the static shared `encoder_hidden_states` input with the layer-specific modulated states $H^{(1..c, l)}$.
2. Apply a block-lower-triangular **Chunk-Causal Cross-Attention Mask** $M_{\text{cross}} \in \{0, -\infty\}^{N_{\text{dec}} \times T'}$ so token $n$ in chunk $c$ attends only to bridge frames $t' \le \text{end}(c)$.
3. Attach an auxiliary **Bridge-CTC Head** to $H^{(0)}$ to produce forced chunk alignments during training and stabilize monotonic timing during streaming inference.

---

## 7. Training Objectives and Procedures

### 7.1 Composite Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seq2seq}} + \lambda_{\text{ctc}}\,\mathcal{L}_{\text{CTC}} + \lambda_{\text{fb}}\,\mathcal{L}_{\text{fb-smooth}}$$

| Loss Term | Description | Default Weight |
| :--- | :--- | :---: |
| **$\mathcal{L}_{\text{seq2seq}}$** | Cross-entropy loss over chunk-interleaved target tokens ending in `<WAIT>` / `<EOS>`. | `1.0` |
| **$\mathcal{L}_{\text{CTC}}$** | Auxiliary CTC loss on $H^{(0)}$ for monotonic alignment and chunk boundary extraction. | `0.3` |
| **$\mathcal{L}_{\text{fb-smooth}}$** | Small L2 regularizer on $\Delta H^{(c, l)} = H^{(c, l)} - H^{(c, 0)}$ to prevent feedback drift on clean utterances. | `0.01` |

### 7.2 Feedback Dropout (Critical Stability Rule)
During training, randomly zero out the feedback vector $u^{(c-1, l)}$ for 20% of training batches (`p_fb_drop = 0.20`). This forces the base bridge $H^{(0)}$ to remain accurate during chunk 1 (cold start) and prevents the decoder from collapsing if an early chunk contains a transcription error.

### 7.3 Step-by-Step Training Procedure (ASD-STE100)

#### Stage 1: Open-Loop Base Bridge Warmup
1. Freeze the causal speech encoder and the pruned `mT5`/`ByT5` decoder.
2. Lock all feedback gates $\alpha_l = 0.0$ (`requires_grad = False`).
3. Train `CausalConvDownsampler`, $\text{FFN}_{\text{base}}$, and the auxiliary Bridge-CTC head for 5,000 steps (`lr = 5e-4`).

#### Stage 2: Closed-Loop Feedback Activation
1. Unlock $\alpha_l$ and all $\text{FFN}_{\text{fb}}^{(l)}$ parameters (`requires_grad = True`).
2. Unfreeze the modified chunk-causal `mT5`/`ByT5` decoder (`lr = 1e-4`) and upper causal encoder layers (`lr = 5e-5`).
3. Enable parallel 2-pass teacher-forced chunk feedback with `p_fb_drop = 0.20` and dynamic chunk sizes `{160 ms, 320 ms, 640 ms}`.
4. Train until balanced multi-domain validation WER converges.

#### Stage 3: Streaming Exposure-Bias Polish (Optional)
1. For the final 2,000 to 5,000 steps, replace teacher-forced $u^{(c-1, l)}$ with **model-predicted** $u^{(c-1, l)}$ on 50% of chunks (scheduled sampling).
2. This closes the train–test gap between gold past tokens and predicted streaming tokens.

---

## 8. Reference PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class CausalConvDownsampler(nn.Module):
    """Strictly causal 1D strided temporal compressor with streaming cache."""

    def __init__(self, dim: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()
        self.pad_len = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None):
        x_t = x.transpose(1, 2)  # (B, D, T)
        x_in = torch.cat([cache, x_t], dim=2) if cache is not None else F.pad(x_t, (self.pad_len, 0))
        new_cache = x_in[:, :, -self.pad_len:].detach()
        out = self.act(self.conv(x_in)).transpose(1, 2)
        return out, new_cache


class FeedbackModulationLayer(nn.Module):
    """Zero-initialized FiLM-bottleneck feedback layer for decoder layer l."""

    def __init__(self, dec_dim: int = 512, fb_rank: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cond_norm = nn.LayerNorm(dec_dim)
        self.cond_proj = nn.Linear(dec_dim, 2 * fb_rank)
        self.down = nn.Linear(dec_dim, fb_rank)
        self.up = nn.Linear(fb_rank, dec_dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # Zero-init gate
        self.out_norm = RMSNorm(dec_dim)

    def forward(
        self,
        h_base: torch.Tensor,
        dec_state_u: torch.Tensor | None,
        fb_drop_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_base:      (B, T'_c, D) base bridge states for chunk c (or full sequence)
            dec_state_u: (B, 1, D) or (B, T'_c, D) causal decoder summary from chunks <= c-1
        Returns:
            h_mod:       (B, T'_c, D) feedback-steered bridge states for cross-attention
            delta_reg:   scalar L2 regularization on feedback delta
        """
        if dec_state_u is None:
            return self.out_norm(h_base), h_base.new_tensor(0.0)

        if self.training and fb_drop_prob > 0.0:
            keep = (torch.rand(h_base.size(0), 1, 1, device=h_base.device) >= fb_drop_prob).to(h_base.dtype)
            dec_state_u = dec_state_u * keep

        cond = self.cond_proj(self.cond_norm(dec_state_u))
        gamma, beta = cond.chunk(2, dim=-1)  # (B, 1 or T'_c, fb_rank)

        mid = (1.0 + gamma) * F.gelu(self.down(h_base)) + beta
        delta = self.up(self.dropout(mid))
        gated_delta = torch.tanh(self.alpha) * delta

        h_mod = self.out_norm(h_base + gated_delta)
        delta_reg = (gated_delta ** 2).mean()
        return h_mod, delta_reg


class ClosedLoopFeedbackBridge(nn.Module):
    """
    Closed-Loop Decoder-Feedback Bridge (CLDF-Bridge).
    Computes base projection H^(0) once, then modulates per feedback-enabled decoder layer l
    using strictly causal past decoder states u^(c-1, l).
    """

    def __init__(
        self,
        enc_dim: int = 512,
        dec_dim: int = 512,
        base_hidden: int = 1024,
        fb_rank: int = 128,
        feedback_layers: tuple[int, ...] = (1, 3, 5),
        downsample_stride: int = 2,
        ema_decay: float = 0.7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feedback_layers = set(feedback_layers)
        self.ema_decay = ema_decay

        self.compressor = CausalConvDownsampler(enc_dim, kernel_size=5, stride=downsample_stride)
        self.base_ffn = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, base_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_hidden, dec_dim),
        )
        self.base_norm = RMSNorm(dec_dim)

        self.fb_modules = nn.ModuleDict({
            str(l): FeedbackModulationLayer(dec_dim=dec_dim, fb_rank=fb_rank, dropout=dropout)
            for l in feedback_layers
        })
        self.init_u = nn.ParameterDict({
            str(l): nn.Parameter(torch.zeros(1, 1, dec_dim))
            for l in feedback_layers
        })

    def encode_base(self, enc_hidden: torch.Tensor, conv_cache: torch.Tensor | None = None):
        v, new_cache = self.compressor(enc_hidden, cache=conv_cache)
        h0 = self.base_norm(self.base_ffn(v))
        return h0, new_cache

    def modulate_for_layer(
        self,
        layer_idx: int,
        h_base: torch.Tensor,
        prev_u: torch.Tensor | None = None,
        fb_drop_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(layer_idx)
        if key not in self.fb_modules:
            return h_base, h_base.new_tensor(0.0)
        u_in = prev_u if prev_u is not None else self.init_u[key].expand(h_base.size(0), 1, -1)
        return self.fb_modules[key](h_base, u_in, fb_drop_prob=fb_drop_prob)

    def update_causal_state(
        self,
        prev_u: torch.Tensor | None,
        chunk_dec_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Causal EMA update after finishing chunk c emission (used for chunk c+1)."""
        if chunk_dec_hidden.size(1) == 0:
            return prev_u
        chunk_mean = chunk_dec_hidden.mean(dim=1, keepdim=True).detach()
        if prev_u is None:
            return chunk_mean
        return self.ema_decay * prev_u + (1.0 - self.ema_decay) * chunk_mean
```

---

## 9. Journal Evaluation Protocol

To satisfy reviewers at **IEEE/ACM TASLP** or **Computer Speech & Language**, execute this evaluation matrix.

### 9.1 Benchmarks (Multilingual, Dialect, and Code-Switch Shifts)
1. **Marathi Dialect Continuum:** IISc RESPIN Marathi (D1 Malvani, D2 Ahirani, D3 Standard, D4 Varhadi) + OOD KathBath Marathi + OOD FLEURS Marathi.
2. **Pan-Indic 9-Language Shift Suite:** All 9 IISc RESPIN languages (`mr, hi, bh, ch, mg, mt, bn, kn, te`) spanning Indo-Aryan continuums, Bengali, and Dravidian scripts.
3. **Intra-Utterance Shift / Code-Switching Stress Test:** Rural Marathi–English and Hindi–Dialect utterances (plus stitched bilingual/bi-dialectal streaming sequences where the speaker switches mid-stream without any pause or Language-ID signal!).

### 9.2 Controlled Baselines (Swap Only the Bridge)
Keep the causal speech encoder and chunk-causal decoder fixed. Compare seven systems:
1. **Causal CTC Baseline** (no Seq2Seq decoder — current Stage 3)
2. **Plain Open-Loop FFN Bridge** (`SLAM-ASR` / `MEUSLI` style)
3. **Q-Former Bridge**
4. **LegoSLM CTC-Posterior Bridge** (2025)
5. **SMEAR-MoE Bridge** (Jan 2026)
6. **Open-Loop FFN + Oracle Language-ID Prompt** (upper bound showing how much LID helps an open-loop model)
7. **CLDF-Bridge (Ours — Zero Language-ID)** — tested with **Indic-Pruned `mT5-small`** and **`ByT5-small`**

### 9.3 Required Ablations and Diagnostic Proofs
1. **Component Ablation Table:**
   - Open-loop ($\alpha_l$ locked to $0$)
   - Chunk-level feedback only (shared single modulation before layer 0) vs. **Layer-wise depth feedback** ($l \in \{1, 3, 5\}$)
   - Remove Feedback Dropout (`p_fb_drop = 0`)
   - EMA decay sweep ($\beta \in \{0.0, 0.5, 0.7, 0.9\}$)
   - `Indic-Pruned mT5-small` vs. `ByT5-small`
2. **Feedback Activation Trajectory Plot (Headline Reviewer Figure):**
   Plot $\|\tanh(\alpha_l)\,\Delta H^{(c, l)}\|_2$ over streaming time $t$ on a mid-utterance dialect or code-switch utterance. Show that feedback modulation spikes cleanly right after a script or dialect shift begins.
3. **Chunk-Position WER Curve (Cold Start vs. Steady State):**
   Plot WER on the first 2 seconds of an utterance vs. seconds 2–10. Show that CLDF-Bridge progressively outperforms open-loop FFN and SMEAR-MoE as causal feedback accumulates.
4. **Streaming Latency–Accuracy Pareto Curve:**
   Report WER, **Streaming RTF**, **Average Lagging (AL)**, and per-chunk compute overhead across `{160 ms, 320 ms, 640 ms, Offline}`.
5. **Statistical Significance:**
   Report speaker-cluster 95% bootstrap confidence intervals and MAPSSWE p-values across all dialect and language slices.

---

## 10. Optional Composition With SOS-Bridge

The `CLDF-Bridge` and the `SOS-Bridge` (defined in `SOS_BRIDGE_SPEC.md`) are mutually orthogonal:
- Replace `base_ffn` inside `ClosedLoopFeedbackBridge` with `SOSBridge` (`W_core + W_shift` + hyperspherical norm calibration).
- Use `CLDF-Bridge` feedback layers to steer the calibrated representation dynamically across chunks.
- Test `SOS-Bridge` alone, `CLDF-Bridge` alone, and `SOS + CLDF` combined in one clean ablation table.

---

## 11. Two-Phase Execution Checklist

### Phase 1: Fast Marathi Streaming Proof-of-Concept (Days 1–5)
1. Prune `mT5-small` vocabulary to Devanagari + Latin (~12k–20k tokens) and add the `<WAIT>` token.
2. Implement the block-lower-triangular chunk-causal cross-attention wrapper on `T5Stack`.
3. Attach `ClosedLoopFeedbackBridge` to your Causal Conformer encoder and run Stage 1 (open-loop warmup) followed by Stage 2 (closed-loop activation) on balanced Marathi data (RESPIN + KathBath + OpenSLR + CommonVoice).
4. Compare `Causal CTC` vs. `Plain Open-Loop FFN` vs. `SMEAR-MoE` vs. `CLDF-Bridge` on RESPIN D1–D4, KathBath, and FLEURS.
5. Verify that `CLDF-Bridge` beats Open-Loop FFN on late-chunk dialect accuracy and code-switched words with less than 5% latency overhead.

### Phase 2: Pan-Indic Multilingual Shift Suite (Days 6–14)
1. Pair `CLDF-Bridge` with a multilingual causal speech encoder (chunked Whisper / MMS / IndicConformer encoder) alongside your Conformer.
2. Train and evaluate across all 9 RESPIN Indic languages (`mr, hi, bh, ch, mg, mt, bn, kn, te`) plus stitched mid-stream language-switch stress tests.
3. Generate the Feedback Activation Trajectory plot, Cold-Start vs. Late-Chunk WER curve, Latency–WER Pareto plot, and bootstrap significance tables.
4. Draft the manuscript for **Computer Speech & Language** or **IEEE/ACM TASLP**.
