"""Core implementation of Medical Terminology Helper (GPT from scratch).

Built milestone by milestone:
  1. Public domain medical terminology corpus curation & loading
  2. Byte-Pair Encoding (BPE) tokenizer & token/positional embeddings
  3. Causal self-attention & Transformer block (MedicalGPT architecture)
  4. Training loop, evaluation, and checkpoint persistence
  5. Autoregressive text generation for medical phrase completions
  6. CLI workflow helpers and interactive study utilities
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "medical_corpus.txt"


# ============================================================================
# Milestone 1: Curate a small corpus of public domain medical texts
# ============================================================================

def load_medical_corpus(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load and parse the curated public-domain medical terminology corpus.

    Returns a list of entries with keys: ``category``, ``term``, and ``text``.
    """
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    if not corpus_path.exists():
        raise FileNotFoundError(f"Medical corpus not found at {corpus_path}")

    entries: list[dict[str, str]] = []
    current_category = "General Medicine"

    for raw_line in corpus_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_category = line[3:].strip()
            continue
        if line.startswith("#"):
            continue

        # Extract leading medical term (e.g. "Myocardium is the..." -> "Myocardium")
        match = re.match(r"^([A-Za-z][A-Za-z\s\-/]+?)\s+(?:is|are|refers|consists|describes|means)\b", line)
        term = match.group(1).strip() if match else line.split()[0].rstrip(".,;:")
        entries.append({
            "category": current_category,
            "term": term,
            "text": line,
        })

    return entries


def get_corpus_text(path: str | Path | None = None) -> str:
    """Return the cleaned, newline-joined training text from the medical corpus."""
    entries = load_medical_corpus(path)
    return "\n".join(entry["text"] for entry in entries)


def search_corpus(query: str, path: str | Path | None = None, limit: int = 5) -> list[dict[str, str]]:
    """Search the curated medical corpus for matching terms or definitions."""
    q = query.strip().lower()
    if not q:
        return []
    entries = load_medical_corpus(path)
    exact_term = [e for e in entries if q == e["term"].lower()]
    prefix_term = [e for e in entries if e not in exact_term and e["term"].lower().startswith(q)]
    contains = [
        e for e in entries
        if e not in exact_term and e not in prefix_term
        and (q in e["term"].lower() or q in e["text"].lower())
    ]
    return (exact_term + prefix_term + contains)[:limit]


def corpus_stats(path: str | Path | None = None) -> dict:
    """Compute summary statistics for the medical terminology corpus."""
    entries = load_medical_corpus(path)
    text = "\n".join(e["text"] for e in entries)
    categories = sorted({e["category"] for e in entries})
    words = text.split()
    return {
        "num_entries": len(entries),
        "num_categories": len(categories),
        "categories": categories,
        "num_words": len(words),
        "num_chars": len(text),
        "sample_terms": [e["term"] for e in entries[:8]],
    }


def milestone_1(path: str | Path | None = None) -> dict:
    """Milestone 1: Load curated public domain medical texts and return corpus data & stats."""
    entries = load_medical_corpus(path)
    text = "\n".join(e["text"] for e in entries)
    stats = corpus_stats(path)
    return {"text": text, "entries": entries, "stats": stats}


# ============================================================================
# Milestone 2: Byte-Pair Encoding (BPE) Tokenizer and Medical Embeddings
# ============================================================================

SPECIAL_TOKENS = {
    "<PAD>": 256,
    "<BOS>": 257,
    "<EOS>": 258,
    "<UNK>": 259,
}
BASE_VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)  # 260


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all non-overlapping occurrences of ``pair`` in ``ids`` with ``new_id``."""
    merged: list[int] = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BytePairTokenizer:
    """Byte-level Byte-Pair Encoding (BPE) tokenizer built from scratch.

    Uses the 256 raw UTF-8 byte values as the base vocabulary so any medical
    term can be encoded losslessly without unknown-token fallback, followed by
    4 special tokens and iteratively learned subword merges.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for tok, idx in SPECIAL_TOKENS.items():
            self.vocab[idx] = tok.encode("utf-8")
        self.pad_id = SPECIAL_TOKENS["<PAD>"]
        self.bos_id = SPECIAL_TOKENS["<BOS>"]
        self.eos_id = SPECIAL_TOKENS["<EOS>"]
        self.unk_id = SPECIAL_TOKENS["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int = 512) -> "BytePairTokenizer":
        """Learn BPE merge rules from ``text`` up to ``vocab_size`` tokens."""
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_VOCAB_SIZE}, got {vocab_size}")

        # Split on whitespace boundaries while preserving leading spaces for natural subwords
        chunks = re.findall(r"\S+\s*|\n+", text)
        token_chunks: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

        num_merges = vocab_size - BASE_VOCAB_SIZE
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for tok, idx in SPECIAL_TOKENS.items():
            self.vocab[idx] = tok.encode("utf-8")

        for step in range(num_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for chunk in token_chunks:
                for i in range(len(chunk) - 1):
                    pair_counts[(chunk[i], chunk[i + 1])] += 1

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < 2:
                break

            new_id = BASE_VOCAB_SIZE + step
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            token_chunks = [_merge_pair(chunk, best_pair, new_id) for chunk in token_chunks]

        return self

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode ``text`` into a sequence of integer token IDs."""
        chunks = re.findall(r"\S+\s*|\n+", text)
        ids: list[int] = [self.bos_id] if add_special_tokens else []

        for chunk in chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                best_pair = None
                best_rank = float("inf")
                for i in range(len(chunk_ids) - 1):
                    pair = (chunk_ids[i], chunk_ids[i + 1])
                    rank = self.merges.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break
                chunk_ids = _merge_pair(chunk_ids, best_pair, int(best_rank))
            ids.extend(chunk_ids)

        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token IDs back into a UTF-8 string."""
        special_ids = set(SPECIAL_TOKENS.values())
        byte_parts: list[bytes] = []
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            byte_parts.append(self.vocab.get(int(idx), b""))
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def tokenize_with_pieces(self, text: str) -> list[dict]:
        """Return token IDs alongside their human-readable decoded pieces."""
        ids = self.encode(text)
        return [
            {
                "id": idx,
                "piece": self.vocab[idx].decode("utf-8", errors="replace"),
                "num_bytes": len(self.vocab[idx]),
            }
            for idx in ids
        ]

    def to_dict(self) -> dict:
        """Serialize tokenizer merges to a JSON-compatible dictionary."""
        return {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BytePairTokenizer":
        """Reconstruct a ``BytePairTokenizer`` from serialized merge rules."""
        tok = cls()
        for a, b, new_id in data.get("merges", []):
            tok.merges[(int(a), int(b))] = int(new_id)
            tok.vocab[int(new_id)] = tok.vocab[int(a)] + tok.vocab[int(b)]
        return tok


class MedicalEmbedding(nn.Module):
    """Combines learned token embeddings and learned positional embeddings."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq_len)`` token IDs to ``(batch, seq_len, d_model)`` embeddings."""
        _, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        return self.dropout(x)


def milestone_2(
    corpus_text: str | None = None,
    vocab_size: int = 400,
    d_model: int = 64,
    max_seq_len: int = 128,
) -> dict:
    """Milestone 2: Train BPE tokenizer on the medical corpus and build embeddings."""
    text = corpus_text if corpus_text is not None else get_corpus_text()
    tokenizer = BytePairTokenizer().train(text, vocab_size=vocab_size)
    embedding_layer = MedicalEmbedding(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    sample_prompt = "Myocardium is the thick muscular middle layer of the heart"
    token_ids = tokenizer.encode(sample_prompt)
    sample_tensor = torch.tensor([token_ids[:max_seq_len]], dtype=torch.long)
    sample_embeddings = embedding_layer(sample_tensor)
    return {
        "tokenizer": tokenizer,
        "embedding_layer": embedding_layer,
        "sample_token_ids": token_ids,
        "sample_embeddings": sample_embeddings,
    }


# ============================================================================
# Milestone 3: Causal Self-Attention, Transformer Block, and MedicalGPT
# ============================================================================

@dataclass
class GPTConfig:
    """Hyperparameters for the from-scratch MedicalGPT model."""

    vocab_size: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    max_seq_len: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GPTConfig":
        return cls(**data)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with upper-triangular future masking."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(d_model, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder Transformer block with causal self-attention and MLP."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MedicalGPT(nn.Module):
    """Decoder-only GPT language model built from scratch in PyTorch."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embedding = MedicalEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Weight tying between token embedding and output projection
        self.lm_head.weight = self.embedding.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def milestone_3(
    vocab_size: int = 400,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    max_seq_len: int = 64,
) -> dict:
    """Milestone 3: Build causal self-attention, transformer block, and full MedicalGPT."""
    config = GPTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    model = MedicalGPT(config)
    model.eval()
    sample_ids = torch.randint(0, vocab_size, (2, 12))
    with torch.no_grad():
        logits, _ = model(sample_ids)
    return {
        "config": config,
        "model": model,
        "block": model.blocks[0],
        "attention": model.blocks[0].attn,
        "sample_logits": logits,
        "num_parameters": model.num_parameters(),
    }


# --- Milestone 4: Set up the training loop and train the GPT model on the medical text data.
# Hint: Training Loop implementation and model training.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement a text generation function to complete medical phrases or definitions.
# Hint: Text Generation function.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Create a CLI for students to enter partial phrases and get completions.
# Hint: Command-Line Interface (CLI) development.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
