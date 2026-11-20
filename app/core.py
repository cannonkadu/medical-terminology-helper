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

from pathlib import Path
import re

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


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the medical corpus.
# Hint: Byte-Pair Tokenization and Embeddings.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


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
