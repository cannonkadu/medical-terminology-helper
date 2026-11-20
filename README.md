# Medical Terminology Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/e24xm4xhp9/badge.svg)](https://ask.thili.ai/b/e24xm4xhp9)

A command-line (and optional local web studio) assistant for medical students that completes medical phrases and definitions using a **decoder-only GPT language model built and trained from scratch in PyTorch** on curated public-domain medical texts.

## Overview

**Goal.** Build a command-line assistant that helps medical students complete medical phrases and definitions.

**Done when.** Successfully apply a byte-pair tokenizer and embeddings to prepare raw medical text, implement causal self-attention, assemble the transformer block, and train a GPT model from scratch to generate coherent medical text completions.

## Milestones

- [x] **1. Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).**  
      Curated 160+ medical definitions, anatomical descriptions, clinical terms, and roots/prefixes/suffixes across 10 medical domains in `corpus/medical_corpus.txt`.
- [x] **2. Implement the byte-pair tokenizer and generate embeddings from the medical corpus.**  
      Implemented a lossless byte-level `BytePairTokenizer` from scratch and `MedicalEmbedding` (learned token + positional embeddings).
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Implemented multi-head `CausalSelfAttention` with upper-triangular masking, `FeedForward` (GELU), Pre-LayerNorm `TransformerBlock`, and weight-tied `MedicalGPT`.
- [x] **4. Set up the training loop and train the GPT model on the medical text data.**  
      Implemented `MedicalTextDataset` (sliding-window next-token prediction), `train_model` with AdamW, cosine LR decay, gradient clipping, train/val loss & perplexity tracking, and checkpoint persistence.
- [x] **5. Implement a text generation function to complete medical phrases or definitions.**  
      Implemented `generate_completion` with autoregressive decoding, temperature scaling, top-k filtering, top-p (nucleus) sampling, and repetition penalty.
- [x] **6. Create a CLI for students to enter partial phrases and get completions.**  
      Built a full CLI (`train`, `complete`, `interactive`, `lookup`, `quiz`, `inspect`, `web`) plus a zero-dependency local Web Studio UI.

## Tech Stack

- **Language:** Python 3.12+
- **Deep Learning:** PyTorch (`torch`) — custom BPE tokenizer, embeddings, multi-head causal self-attention, transformer blocks, and training loop built from scratch
- **CLI & Web Studio:** Python standard library (`argparse`, `http.server`)
- **Testing & CI:** `pytest` + GitHub Actions (`.github/workflows/ci.yml`)
- **Optional LLM Tutor:** OpenAI-compatible chat client (`httpx`) supporting Ollama, Groq, or Gemini via `.env`

## Project Structure

```text
medical-terminology-helper/
├── .env.example              # Optional LLM provider settings (never commit .env)
├── .github/workflows/ci.yml  # GitHub Actions CI running pytest
├── corpus/
│   └── medical_corpus.txt    # Curated public-domain medical terminology corpus
├── app/
│   ├── __init__.py
│   ├── core.py               # Milestones 1–6: Corpus, BPE, MedicalGPT, training, generation
│   ├── cli.py                # Command-line interface (`python -m app.cli`)
│   ├── web.py                # Local Web Studio UI (`python -m app.cli web`)
│   ├── llm.py                # Optional LLM tutor integration (Ollama / Groq / Gemini)
│   └── storage.py            # Lightweight JSON key-value store
├── tests/
│   ├── test_smoke.py         # Plumbing smoke tests
│   └── test_core.py          # End-to-end tests for Milestones 1–6
├── pytest.ini
└── requirements.txt
```

## Quick Start

### 1. Set up the environment

```bash
python3.12 -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Verify installation & run tests

```bash
python -m app.cli hello
pytest
```

### 3. Train MedicalGPT from scratch

```bash
python -m app.cli train --epochs 25
```

This trains the byte-pair tokenizer and 3-layer causal Transformer (`MedicalGPT`) on `corpus/medical_corpus.txt` and saves the checkpoint to `data/medical_gpt.pt`.

### 4. Complete medical phrases from the CLI

```bash
python -m app.cli complete "Myocardium is " --reference
python -m app.cli complete "Alveoli are " --temperature 0.5 --max-tokens 32
python -m app.cli complete "The suffix itis "
```

### 5. Explore interactive study tools

```bash
# Interactive study REPL (/lookup, /quiz, /temp, /tokens, /stats, /quit)
python -m app.cli interactive

# Practice with fill-in-the-blank terminology quizzes
python -m app.cli quiz --count 5

# Search the curated medical corpus
python -m app.cli lookup bradycardia

# Inspect BPE tokenization & model architecture
python -m app.cli inspect --text "Myocardial infarction is the irreversible necrosis of heart muscle tissue."

# Launch the local Web Studio UI (http://127.0.0.1:8000)
python -m app.cli web
```

## Settings and Secrets (Optional LLM Tutor)

Everything works **100% offline on your laptop** using the from-scratch PyTorch GPT model. If you also want optional AI tutor explanations with `complete --explain`, copy `.env.example` to `.env` and configure Ollama, Groq, or Gemini:

```bash
cp .env.example .env
```

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.**
