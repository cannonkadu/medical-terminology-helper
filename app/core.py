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


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal Self-Attention and Transformer Block assembly.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


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
