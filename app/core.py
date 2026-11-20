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
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "medical_corpus.txt"
DEFAULT_CHECKPOINT_PATH = Path("data/medical_gpt.pt")


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


# ============================================================================
# Milestone 4: Dataset, Training Loop, and Model Checkpointing
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training MedicalGPT from scratch on the medical corpus."""

    vocab_size: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    max_seq_len: int = 96
    dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 25
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    stride: int = 16
    seed: int = 42
    device: str = "auto"


class MedicalTextDataset(Dataset):
    """Sliding-window next-token prediction dataset over tokenized medical text."""

    def __init__(self, token_ids: list[int], seq_len: int, stride: int = 16):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        if len(token_ids) < 2:
            raise ValueError("Need at least 2 tokens to build training pairs.")

        effective_len = min(seq_len, len(token_ids) - 1)
        step = max(1, min(stride, effective_len))

        for start in range(0, len(token_ids) - effective_len, step):
            x = torch.tensor(token_ids[start : start + effective_len], dtype=torch.long)
            y = torch.tensor(token_ids[start + 1 : start + effective_len + 1], dtype=torch.long)
            self.samples.append((x, y))

        # Ensure the tail of the corpus is always included
        last_start = len(token_ids) - effective_len - 1
        if not self.samples or last_start > (len(self.samples) - 1) * step:
            x = torch.tensor(token_ids[last_start : last_start + effective_len], dtype=torch.long)
            y = torch.tensor(token_ids[last_start + 1 : last_start + effective_len + 1], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(
    path: str | Path,
    model: MedicalGPT,
    tokenizer: BytePairTokenizer,
    history: list[dict] | None = None,
) -> Path:
    """Persist trained model weights, config, tokenizer merges, and training history."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "history": history or [],
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    device: str = "auto",
) -> tuple[MedicalGPT, BytePairTokenizer, list[dict]]:
    """Load a saved MedicalGPT model and its BPE tokenizer from disk."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    resolved = _resolve_device(device)
    payload = torch.load(ckpt_path, map_location=resolved, weights_only=False)
    config = GPTConfig.from_dict(payload["config"])
    model = MedicalGPT(config).to(resolved)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    tokenizer = BytePairTokenizer.from_dict(payload["tokenizer"])
    history = payload.get("history", [])
    return model, tokenizer, history


def train_model(
    corpus_text: str | None = None,
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Train MedicalGPT from scratch on the curated medical text corpus."""
    cfg = config or TrainingConfig()
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    text = corpus_text if corpus_text is not None else get_corpus_text()
    tokenizer = BytePairTokenizer().train(text, vocab_size=cfg.vocab_size)
    token_ids = tokenizer.encode(text)

    # 90/10 train/validation split when corpus is large enough
    split_idx = int(len(token_ids) * 0.9)
    if len(token_ids) - split_idx > cfg.max_seq_len + 1:
        train_ids = token_ids[:split_idx]
        val_ids = token_ids[split_idx:]
    else:
        train_ids = token_ids
        val_ids = token_ids

    train_dataset = MedicalTextDataset(train_ids, seq_len=cfg.max_seq_len, stride=cfg.stride)
    val_dataset = MedicalTextDataset(val_ids, seq_len=cfg.max_seq_len, stride=max(cfg.stride, cfg.max_seq_len // 2))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    )
    model = MedicalGPT(gpt_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
        eta_min=cfg.learning_rate * 0.1,
    )

    history: list[dict] = []
    start_time = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train_loss = 0.0
        batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x_batch, y_batch)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_train_loss += float(loss.item())
            batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, batches)

        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                _, v_loss = model(x_val, y_val)
                assert v_loss is not None
                total_val_loss += float(v_loss.item())
                val_batches += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        perplexity = math.exp(min(avg_val_loss, 20.0))

        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "perplexity": round(perplexity, 2),
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

    elapsed = round(time.perf_counter() - start_time, 2)
    model.eval()

    saved_path = None
    if checkpoint_path is not None:
        saved_path = str(save_checkpoint(checkpoint_path, model, tokenizer, history))

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": gpt_config,
        "history": history,
        "initial_loss": history[0]["train_loss"],
        "final_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_perplexity": history[-1]["perplexity"],
        "num_tokens": len(token_ids),
        "num_samples": len(train_dataset),
        "num_parameters": model.num_parameters(),
        "elapsed_seconds": elapsed,
        "checkpoint_path": saved_path,
    }


def milestone_4(
    corpus_text: str | None = None,
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Milestone 4: Train MedicalGPT on the medical text corpus and verify loss decreases."""
    return train_model(
        corpus_text=corpus_text,
        config=config,
        checkpoint_path=checkpoint_path,
    )


# ============================================================================
# Milestone 5: Autoregressive Text Generation for Medical Phrase Completions
# ============================================================================

@torch.no_grad()
def generate_completion(
    model: MedicalGPT,
    tokenizer: BytePairTokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 25,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stop_on_newline: bool = True,
) -> dict:
    """Autoregressively complete a partial medical phrase or definition."""
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]

    context = list(prompt_ids)
    generated_ids: list[int] = []
    banned_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id}

    for _ in range(max_new_tokens):
        window = context[-model.config.max_seq_len :]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1, :].clone()

        # Suppress non-text special tokens
        for bad_id in banned_ids:
            if bad_id < next_logits.size(0):
                next_logits[bad_id] = float("-inf")

        # Mild repetition penalty over recently generated tokens
        if repetition_penalty > 1.0 and generated_ids:
            for prev_id in set(generated_ids[-16:]):
                if next_logits[prev_id] > 0:
                    next_logits[prev_id] /= repetition_penalty
                else:
                    next_logits[prev_id] *= repetition_penalty

        if temperature <= 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / temperature

            if top_k > 0 and top_k < next_logits.size(0):
                top_vals, _ = torch.topk(next_logits, top_k)
                cutoff = top_vals[-1]
                next_logits = torch.where(
                    next_logits < cutoff,
                    torch.tensor(float("-inf"), device=device),
                    next_logits,
                )

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = False
                remove_indices = sorted_indices[sorted_mask]
                next_logits[remove_indices] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id == tokenizer.eos_id:
            break

        generated_ids.append(next_id)
        context.append(next_id)

        if stop_on_newline:
            decoded_piece = tokenizer.decode([next_id])
            if "\n" in decoded_piece:
                break

    completion = tokenizer.decode(generated_ids)
    if stop_on_newline and "\n" in completion:
        completion = completion.split("\n", 1)[0]

    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": f"{prompt}{completion}",
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
    }


def ensure_trained_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    config: TrainingConfig | None = None,
    force_retrain: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[MedicalGPT, BytePairTokenizer, list[dict]]:
    """Load an existing MedicalGPT checkpoint, or train one from scratch if missing."""
    ckpt = Path(checkpoint_path)
    if ckpt.exists() and not force_retrain:
        return load_checkpoint(ckpt)
    result = train_model(
        config=config or TrainingConfig(),
        checkpoint_path=ckpt,
        progress_callback=progress_callback,
    )
    return result["model"], result["tokenizer"], result["history"]


def milestone_5(
    prompt: str = "Myocardium is ",
    model: MedicalGPT | None = None,
    tokenizer: BytePairTokenizer | None = None,
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.9,
) -> dict:
    """Milestone 5: Generate medical phrase or definition completions from a trained GPT."""
    if model is None or tokenizer is None:
        cfg = TrainingConfig(
            vocab_size=360,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=64,
            batch_size=16,
            epochs=10,
            learning_rate=5e-3,
            stride=20,
            device="cpu",
        )
        trained = train_model(config=cfg)
        model = trained["model"]
        tokenizer = trained["tokenizer"]

    return generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


# --- Milestone 6: Create a CLI for students to enter partial phrases and get completions.
# Hint: Command-Line Interface (CLI) development.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
