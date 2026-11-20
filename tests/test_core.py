"""Tests for Medical Terminology Helper milestones."""
import pytest

from app.core import (
    BytePairTokenizer,
    MedicalEmbedding,
    corpus_stats,
    get_corpus_text,
    load_medical_corpus,
    milestone_1,
    milestone_2,
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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Set up the training loop and train the GPT model on the medical text data.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement a text generation function to complete medical phrases or definitions.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a CLI for students to enter partial phrases and get completions.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
