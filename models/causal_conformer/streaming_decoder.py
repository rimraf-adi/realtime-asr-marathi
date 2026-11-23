"""
Incremental Streaming CTC Decoder for Real-Time ASR.
Maintains decoding state (last predicted token, blank status, committed and partial hypotheses)
across consecutive audio chunks without resetting at chunk boundaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch


@dataclass
class DecoderState:
    """State preserved across consecutive streaming chunks."""
    last_token_id: int = 0
    tokens: List[int] = field(default_factory=list)
    committed_text: str = ""
    partial_text: str = ""

    def reset(self, blank_id: int = 0) -> None:
        self.last_token_id = blank_id
        self.tokens.clear()
        self.committed_text = ""
        self.partial_text = ""


class StreamingCTCDecoder:
    """
    Incremental CTC Greedy & Prefix Decoder.
    Guarantees consistent CTC collapse across streaming chunk boundaries.
    """

    def __init__(
        self,
        vocab_dict: Dict[str, Any],
        blank_id: int = 0,
        commit_lag_frames: int = 2,
    ):
        self.blank_id = blank_id
        self.commit_lag_frames = commit_lag_frames

        # Build id2token mapping
        token2id = vocab_dict.get("token2id", vocab_dict)
        self.id2token: Dict[int, str] = {
            int(idx): str(tok) for tok, idx in token2id.items()
        }
        self.special_ids = {
            vocab_dict.get("blank_id", 0),
            vocab_dict.get("unk_id", 1),
            vocab_dict.get("pad_id", 2),
        }

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Decode a list of token IDs into text string."""
        chars = []
        for t in tokens:
            if t not in self.special_ids and t in self.id2token:
                tok = self.id2token[t]
                if tok == "|":
                    chars.append(" ")
                elif not tok.startswith("<") and not tok.startswith("["):
                    chars.append(tok)
        return "".join(chars)

    def init_state(self) -> DecoderState:
        """Create a new decoder state."""
        return DecoderState(last_token_id=self.blank_id)

    def decode_chunk(
        self,
        logits: torch.Tensor,
        state: Optional[DecoderState] = None,
        is_final: bool = False,
    ) -> Tuple[str, DecoderState]:
        """
        Incrementally decode a chunk of CTC logits.
        Args:
            logits: Tensor of shape (time_frames, vocab_size) or (1, time_frames, vocab_size)
            state: DecoderState from previous chunk (or None to initialize)
            is_final: If True, flush all remaining partial tokens as committed
        Returns:
            hyp_text: Current cumulative transcription text (committed + partial)
            state: Updated DecoderState to pass to the next chunk
        """
        if state is None:
            state = self.init_state()

        if logits.dim() == 3:
            logits = logits.squeeze(0)  # (time, vocab_size)

        preds = torch.argmax(logits, dim=-1).tolist()  # List of token IDs

        new_tokens = []
        last_id = state.last_token_id

        for token in preds:
            if token == self.blank_id:
                last_id = self.blank_id
            elif token != last_id:
                new_tokens.append(token)
                last_id = token

        state.last_token_id = last_id
        state.tokens.extend(new_tokens)

        if is_final:
            state.committed_text = self.tokens_to_text(state.tokens)
            state.partial_text = ""
            return state.committed_text, state

        # Determine committed vs partial boundary
        if len(state.tokens) > self.commit_lag_frames:
            committed_tokens = state.tokens[:-self.commit_lag_frames]
            partial_tokens = state.tokens[-self.commit_lag_frames:]
        else:
            committed_tokens = []
            partial_tokens = state.tokens

        state.committed_text = self.tokens_to_text(committed_tokens)
        state.partial_text = self.tokens_to_text(partial_tokens)

        # Full hypothesis text
        full_text = self.tokens_to_text(state.tokens)
        return full_text, state
