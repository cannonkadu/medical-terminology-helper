# Medical Terminology Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/e24xm4xhp9/badge.svg)](https://ask.thili.ai/b/e24xm4xhp9)

This project delivers a command-line assistant for medical students, designed to complete medical phrases and definitions. It leverages a GPT model trained on public domain medical texts, reinforcing terminology learning by predicting likely next words in a medical context.

## What you are building

**Goal.** You will build a command-line assistant that helps medical students complete medical phrases and definitions.

**Done when.** You will have successfully applied byte-pair tokenizer and embeddings to prepare raw medical text, implemented causal self-attention, assembled the transformer block, and trained a GPT model from scratch to generate coherent medical text completions.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).**  
      Hint: Data preparation and corpus selection.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the medical corpus.**  
      Hint: Byte-Pair Tokenization and Embeddings.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal Self-Attention and Transformer Block assembly.
- [ ] **4. Set up the training loop and train the GPT model on the medical text data.**  
      Hint: Training Loop implementation and model training.
- [ ] **5. Implement a text generation function to complete medical phrases or definitions.**  
      Hint: Text Generation function.
- [ ] **6. Create a CLI for students to enter partial phrases and get completions.**  
      Hint: Command-Line Interface (CLI) development.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
