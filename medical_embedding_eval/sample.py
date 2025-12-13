"""Data structures for medical samples and their variations."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class MedicalSample:
    """Represents a medical record sample.
    
    Attributes:
        text: The original medical text
        sample_id: Unique identifier for the sample
        metadata: Additional information about the sample (e.g., domain, language)
        created_at: Timestamp of sample creation
    """
    text: str
    sample_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate sample data."""
        if not self.text or not self.text.strip():
            raise ValueError("Sample text cannot be empty")
        if not self.sample_id:
            raise ValueError("Sample ID is required")


@dataclass
class SampleVariation:
    """Represents a semantic variation of a medical sample.
    
    This class holds variations that preserve semantic meaning while changing:
    - Phrasing and word order
    - Synonyms and alternative terms
    - Medical brands, procedures, diagnoses
    - Sentence structure
    
    Attributes:
        original_sample: Reference to the original medical sample
        variation_text: The varied version of the text
        variation_type: Type of variation applied (e.g., 'paraphrase', 'synonym', 'medical_term')
        variation_id: Unique identifier for this variation
        changes_applied: Description of changes made
        metadata: Additional metadata about the variation
    """
    original_sample: MedicalSample
    variation_text: str
    variation_type: str
    variation_id: str
    changes_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_label: Optional[float] = None
    
    def __post_init__(self):
        """Validate variation data."""
        if not self.variation_text or not self.variation_text.strip():
            raise ValueError("Variation text cannot be empty")
        if not self.variation_id:
            raise ValueError("Variation ID is required")
        
        # Ensure variation is different from original
        if self.variation_text.strip() == self.original_sample.text.strip():
            raise ValueError("Variation must differ from original sample")

        if self.similarity_label is not None:
            if not isinstance(self.similarity_label, (int, float)):
                raise ValueError("similarity_label must be numeric")
            if not 0.0 <= float(self.similarity_label) <= 1.0:
                raise ValueError("similarity_label must be between 0 and 1")
            self.similarity_label = float(self.similarity_label)


@dataclass
class SamplePair:
    """Represents a pair of original sample and its variation for evaluation.
    
    Attributes:
        original: The original medical sample
        variation: The semantic variation
        expected_similarity: Expected similarity score (optional, for validation)
    """
    original: MedicalSample
    variation: SampleVariation
    expected_similarity: Optional[float] = None
    
    def __post_init__(self):
        """Validate that variation belongs to the original."""
        if self.variation.original_sample != self.original:
            raise ValueError("Variation must belong to the original sample")
