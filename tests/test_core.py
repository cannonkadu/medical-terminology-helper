"""Tests for Medical Terminology Helper milestones."""
import pytest
import torch

from app.core import (
    BytePairTokenizer,
    CausalSelfAttention,
    GPTConfig,
    MedicalEmbedding,
    MedicalGPT,
    TrainingConfig,
    TransformerBlock,
    corpus_stats,
    generate_completion,
    get_corpus_text,
    load_checkpoint,
    load_medical_corpus,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    search_corpus,
)


# Milestone 1: Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).
def test_milestone_1():
    result = milestone_1()
    assert len(result["entries"]) >= 100
    assert result["stats"]["num_categories"] >= 8
    assert result["stats"]["num_words"] >= 1500
    assert "Myocardium" in result["text"]
    assert "Alveoli" in result["text"]
    assert "Neuron" in result["text"]

    matches = search_corpus("myocardium")
    assert len(matches) >= 1
    assert matches[0]["term"].lower() == "myocardium"


# Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the medical corpus.
def test_milestone_2():
    result = milestone_2(vocab_size=360, d_model=32, max_seq_len=64)
    tokenizer = result["tokenizer"]
    embeddings = result["sample_embeddings"]

    assert isinstance(tokenizer, BytePairTokenizer)
    assert tokenizer.vocab_size > 260
    assert tokenizer.vocab_size <= 360

    phrase = "Bradycardia is an abnormally slow resting heart rate."
    ids = tokenizer.encode(phrase)
    assert len(ids) < len(phrase.encode("utf-8"))  # BPE compresses medical text
    assert tokenizer.decode(ids) == phrase

    # Serialization round-trip
    restored = BytePairTokenizer.from_dict(tokenizer.to_dict())
    assert restored.encode(phrase) == ids
    assert restored.decode(ids) == phrase

    # Embedding shape check: (batch=1, seq_len, d_model=32)
    assert embeddings.shape == (1, len(result["sample_token_ids"]), 32)


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
def test_milestone_3():
    result = milestone_3(vocab_size=320, d_model=64, n_heads=4, n_layers=2, max_seq_len=64)
    model = result["model"]
    assert isinstance(model, MedicalGPT)
    assert isinstance(result["block"], TransformerBlock)
    assert isinstance(result["attention"], CausalSelfAttention)
    assert result["sample_logits"].shape == (2, 12, 320)
    assert result["num_parameters"] > 0

    # Strict causality check: altering future tokens must NOT alter earlier logits
    model.eval()
    seq_a = torch.tensor([[10, 20, 30, 40, 50, 60]])
    seq_b = torch.tensor([[10, 20, 30, 99, 88, 77]])
    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)
    assert torch.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-5)
    assert not torch.allclose(logits_a[:, 3:, :], logits_b[:, 3:, :], atol=1e-5)


# Milestone 4: Set up the training loop and train the GPT model on the medical text data.
def test_milestone_4(tmp_path):
    ckpt = tmp_path / "medical_gpt.pt"
    cfg = TrainingConfig(
        vocab_size=340,
        d_model=48,
        n_heads=4,
        n_layers=2,
        max_seq_len=48,
        batch_size=16,
        epochs=6,
        learning_rate=5e-3,
        stride=24,
        device="cpu",
    )
    result = milestone_4(config=cfg, checkpoint_path=ckpt)

    assert len(result["history"]) == 6
    assert result["final_loss"] < result["initial_loss"]
    assert ckpt.exists()

    # Verify checkpoint round-trip loading
    loaded_model, loaded_tok, loaded_history = load_checkpoint(ckpt, device="cpu")
    assert loaded_model.config.d_model == 48
    assert loaded_tok.vocab_size == result["tokenizer"].vocab_size
    assert len(loaded_history) == 6


# Milestone 5: Implement a text generation function to complete medical phrases or definitions.
def test_milestone_5():
    result = milestone_5(
        prompt="Myocardium is ",
        max_new_tokens=20,
        temperature=0.0,  # deterministic greedy decoding
    )
    assert result["prompt"] == "Myocardium is "
    assert result["full_text"].startswith("Myocardium is ")
    assert result["generated_tokens"] > 0
    assert isinstance(result["completion"], str)
    assert len(result["completion"].strip()) > 0


# Milestone 6: Create a CLI for students to enter partial phrases and get completions.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
