"""Tests for sample data structures."""

import pytest
from datetime import datetime

from medical_embedding_eval.sample import MedicalSample, SampleVariation, SamplePair


class TestMedicalSample:
    """Tests for MedicalSample class."""
    
    def test_create_valid_sample(self):
        """Test creating a valid medical sample."""
        sample = MedicalSample(
            text="Pasienten har høyt blodtrykk",
            sample_id="sample_001",
            metadata={"language": "norwegian"}
        )
        assert sample.text == "Pasienten har høyt blodtrykk"
        assert sample.sample_id == "sample_001"
        assert sample.metadata["language"] == "norwegian"
        assert isinstance(sample.created_at, datetime)
    
    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            MedicalSample(text="", sample_id="sample_001")
    
    def test_whitespace_only_text_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            MedicalSample(text="   ", sample_id="sample_001")
    
    def test_missing_sample_id_raises_error(self):
        """Test that missing sample ID raises ValueError."""
        with pytest.raises(ValueError, match="Sample ID is required"):
            MedicalSample(text="Some text", sample_id="")


class TestSampleVariation:
    """Tests for SampleVariation class."""
    
    def test_create_valid_variation(self):
        """Test creating a valid variation."""
        original = MedicalSample(
            text="Pasienten har høyt blodtrykk",
            sample_id="sample_001"
        )
        variation = SampleVariation(
            original_sample=original,
            variation_text="Personen har forhøyet blodtrykksmåling",
            variation_type="paraphrase",
            variation_id="var_001",
            changes_applied=["Replaced 'pasienten' with 'personen'"]
        )
        assert variation.variation_text == "Personen har forhøyet blodtrykksmåling"
        assert variation.variation_type == "paraphrase"
        assert variation.original_sample == original
    
    def test_empty_variation_text_raises_error(self):
        """Test that empty variation text raises ValueError."""
        original = MedicalSample(
            text="Pasienten har høyt blodtrykk",
            sample_id="sample_001"
        )
        with pytest.raises(ValueError, match="Variation text cannot be empty"):
            SampleVariation(
                original_sample=original,
                variation_text="",
                variation_type="paraphrase",
                variation_id="var_001"
            )
    
    def test_identical_variation_raises_error(self):
        """Test that variation identical to original raises ValueError."""
        original = MedicalSample(
            text="Pasienten har høyt blodtrykk",
            sample_id="sample_001"
        )
        with pytest.raises(ValueError, match="Variation must differ from original"):
            SampleVariation(
                original_sample=original,
                variation_text="Pasienten har høyt blodtrykk",
                variation_type="paraphrase",
                variation_id="var_001"
            )


class TestSamplePair:
    """Tests for SamplePair class."""
    
    def test_create_valid_pair(self):
        """Test creating a valid sample pair."""
        original = MedicalSample(
            text="Pasienten har høyt blodtrykk",
            sample_id="sample_001"
        )
        variation = SampleVariation(
            original_sample=original,
            variation_text="Personen har forhøyet blodtrykksmåling",
            variation_type="paraphrase",
            variation_id="var_001"
        )
        pair = SamplePair(original=original, variation=variation)
        assert pair.original == original
        assert pair.variation == variation
        assert pair.expected_similarity is None
    
    def test_mismatched_pair_raises_error(self):
        """Test that mismatched variation raises ValueError."""
        original1 = MedicalSample(
            text="Pasienten har høyt blodtrykk",
            sample_id="sample_001"
        )
        original2 = MedicalSample(
            text="Pasienten har diabetes",
            sample_id="sample_002"
        )
        variation = SampleVariation(
            original_sample=original2,
            variation_text="Personen har sukkersyke",
            variation_type="paraphrase",
            variation_id="var_001"
        )
        with pytest.raises(ValueError, match="Variation must belong to the original"):
            SamplePair(original=original1, variation=variation)
