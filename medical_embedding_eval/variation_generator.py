"""Utilities for generating semantic variations of medical samples.

This module provides tools to create variations of medical text that preserve
semantic meaning while changing surface form (paraphrasing, synonyms, etc.).
"""

from typing import List, Dict, Optional
from .sample import MedicalSample, SampleVariation


class VariationGenerator:
    """Base class for generating semantic variations.
    
    This is a template for implementing variation strategies.
    Users can extend this class or provide their own variations manually.
    """
    
    def generate_variation(self,
                          sample: MedicalSample,
                          variation_type: str,
                          variation_id: Optional[str] = None) -> SampleVariation:
        """Generate a single variation of a sample.
        
        This is a template method to be implemented by subclasses.
        
        Args:
            sample: Original medical sample
            variation_type: Type of variation to generate
            variation_id: Optional ID for the variation (auto-generated if not provided)
            
        Returns:
            SampleVariation object
        """
        raise NotImplementedError("Subclasses must implement generate_variation")
    
    def generate_variations(self,
                          sample: MedicalSample,
                          num_variations: int = 1,
                          variation_types: Optional[List[str]] = None) -> List[SampleVariation]:
        """Generate multiple variations of a sample.
        
        Args:
            sample: Original medical sample
            num_variations: Number of variations to generate
            variation_types: Optional list of variation types to use
            
        Returns:
            List of SampleVariation objects
        """
        variations = []
        for i in range(num_variations):
            var_type = variation_types[i % len(variation_types)] if variation_types else "default"
            variation_id = f"{sample.sample_id}_var_{i+1}"
            variation = self.generate_variation(sample, var_type, variation_id)
            variations.append(variation)
        return variations


class ManualVariationGenerator(VariationGenerator):
    """Helper for creating variations from manually provided texts.
    
    This is useful when variations are created by humans or external tools
    (e.g., manual paraphrasing, translation, or LLM-based generation).
    """
    
    @staticmethod
    def create_variation(original_sample: MedicalSample,
                        variation_text: str,
                        variation_type: str,
                        variation_id: Optional[str] = None,
                        changes_applied: Optional[List[str]] = None) -> SampleVariation:
        """Create a variation from provided text.
        
        Args:
            original_sample: Original medical sample
            variation_text: The varied text
            variation_type: Type of variation (e.g., 'paraphrase', 'synonym_replacement')
            variation_id: Optional ID (auto-generated if not provided)
            changes_applied: Optional list of changes made
            
        Returns:
            SampleVariation object
        """
        if variation_id is None:
            variation_id = f"{original_sample.sample_id}_manual_var"
        
        if changes_applied is None:
            changes_applied = []
        
        return SampleVariation(
            original_sample=original_sample,
            variation_text=variation_text,
            variation_type=variation_type,
            variation_id=variation_id,
            changes_applied=changes_applied,
        )


class SynonymReplacementExample:
    """Example implementation showing how to create variations.
    
    This is a simple example for demonstration purposes only.
    It uses basic string operations and has known limitations:
    - Simple substring matching without word boundaries
    - Case-sensitive replacement
    - No linguistic analysis
    
    Real implementations should use:
    - NLP libraries (spaCy, NLTK) for proper tokenization
    - Medical ontologies (SNOMED CT, ICD codes)
    - LLMs for sophisticated paraphrasing
    - Word boundary detection for accurate replacement
    """
    
    # Example Norwegian medical synonyms/alternatives
    NORWEGIAN_MEDICAL_SYNONYMS = {
        "pasient": ["person", "bruker", "klient"],
        "diagnose": ["sykdom", "tilstand", "lidelse"],
        "medisin": ["medikament", "legemiddel", "preparat"],
        "behandling": ["terapi", "kurering", "behandlingsmetode"],
        "symptom": ["tegn", "plage", "symptomer"],
        "undersøkelse": ["kontroll", "vurdering", "gjennomgang"],
        "smerte": ["pine", "vondt", "smerter"],
        "blodtrykk": ["blodtrykksmåling", "trykk", "BT"],
    }
    
    @classmethod
    def replace_with_synonym(cls, 
                           text: str, 
                           word: str, 
                           replacement: str) -> str:
        """Simple word replacement in text.
        
        Args:
            text: Original text
            word: Word to replace
            replacement: Replacement word
            
        Returns:
            Text with replacement applied
        """
        # Simple case-sensitive replacement
        # In production, use more sophisticated NLP tools
        return text.replace(word, replacement)
    
    @classmethod
    def generate_synonym_variation(cls,
                                   sample: MedicalSample,
                                   variation_id: Optional[str] = None) -> Optional[SampleVariation]:
        """Generate a variation by replacing a word with a synonym.
        
        Args:
            sample: Original medical sample
            variation_id: Optional variation ID
            
        Returns:
            SampleVariation if a synonym was found and replaced, None otherwise
        """
        text = sample.text
        
        # Find first word that has synonyms
        for word, synonyms in cls.NORWEGIAN_MEDICAL_SYNONYMS.items():
            if word in text.lower():
                # Use first synonym
                replacement = synonyms[0]
                new_text = cls.replace_with_synonym(text, word, replacement)
                
                if variation_id is None:
                    variation_id = f"{sample.sample_id}_syn_var"
                
                return SampleVariation(
                    original_sample=sample,
                    variation_text=new_text,
                    variation_type="synonym_replacement",
                    variation_id=variation_id,
                    changes_applied=[f"Replaced '{word}' with '{replacement}'"],
                    metadata={"original_word": word, "synonym": replacement},
                )
        
        return None
