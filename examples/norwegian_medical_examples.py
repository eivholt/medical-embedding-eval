"""Norwegian medical text examples for evaluation.

This file contains example Norwegian medical texts with their semantic variations.
These examples demonstrate various types of variations while preserving semantic meaning.
"""

from medical_embedding_eval import MedicalSample, SampleVariation


def get_norwegian_medical_samples():
    """Get a collection of Norwegian medical sample texts.
    
    Returns:
        List of MedicalSample objects
    """
    samples = [
        MedicalSample(
            text="Pasienten har høyt blodtrykk og diabetes type 2.",
            sample_id="no_001",
            metadata={
                "language": "norwegian",
                "domain": "cardiology/endocrinology",
                "type": "diagnosis"
            }
        ),
        MedicalSample(
            text="Undersøkelsen viser symptomer på lungebetennelse med feber og hoste.",
            sample_id="no_002",
            metadata={
                "language": "norwegian",
                "domain": "pulmonology",
                "type": "examination"
            }
        ),
        MedicalSample(
            text="Pasienten rapporterer sterke smerter i høyre kne etter fall.",
            sample_id="no_003",
            metadata={
                "language": "norwegian",
                "domain": "orthopedics",
                "type": "complaint"
            }
        ),
        MedicalSample(
            text="Behandling med Paracet 500mg tre ganger daglig i fem dager.",
            sample_id="no_004",
            metadata={
                "language": "norwegian",
                "domain": "pharmacy",
                "type": "prescription"
            }
        ),
        MedicalSample(
            text="Blodprøven viser forhøyede verdier av kolesterol og triglyserider.",
            sample_id="no_005",
            metadata={
                "language": "norwegian",
                "domain": "laboratory",
                "type": "test_result"
            }
        ),
        MedicalSample(
            text="Pasienten har allergi mot pollen og får Zyrtec i pollensesongen.",
            sample_id="no_006",
            metadata={
                "language": "norwegian",
                "domain": "allergy",
                "type": "treatment"
            }
        ),
    ]
    return samples


def get_norwegian_medical_variations():
    """Get semantic variations for Norwegian medical samples.
    
    Returns:
        List of SampleVariation objects
    """
    samples = get_norwegian_medical_samples()
    
    variations = [
        # Variation 1: Paraphrase with synonym replacement
        SampleVariation(
            original_sample=samples[0],
            variation_text="Personen lider av forhøyet blodtrykksmåling og sukkersyke type 2.",
            variation_type="paraphrase_with_synonyms",
            variation_id="no_var_001",
            changes_applied=[
                "Replaced 'pasienten har' with 'personen lider av'",
                "Replaced 'høyt blodtrykk' with 'forhøyet blodtrykksmåling'",
                "Replaced 'diabetes' with 'sukkersyke'"
            ],
            metadata={"difficulty": "medium"}
        ),
        
        # Variation 2: Medical term variation
        SampleVariation(
            original_sample=samples[1],
            variation_text="Kontrollen indikerer tegn på pneumoni med pyreksi og hosterefleks.",
            variation_type="medical_term_variation",
            variation_id="no_var_002",
            changes_applied=[
                "Replaced 'undersøkelsen' with 'kontrollen'",
                "Replaced 'symptomer' with 'tegn'",
                "Replaced 'lungebetennelse' with 'pneumoni'",
                "Replaced 'feber' with 'pyreksi'",
                "Replaced 'hoste' with 'hosterefleks'"
            ],
            metadata={"difficulty": "hard"}
        ),
        
        # Variation 3: Word order and synonym changes
        SampleVariation(
            original_sample=samples[2],
            variation_text="Det rapporteres kraftige plager i det høyre kneet hos personen etter et fall.",
            variation_type="word_order_and_synonyms",
            variation_id="no_var_003",
            changes_applied=[
                "Changed word order (passive construction)",
                "Replaced 'sterke smerter' with 'kraftige plager'",
                "Replaced 'pasienten' with 'personen'",
                "Added article 'et' before 'fall'"
            ],
            metadata={"difficulty": "medium"}
        ),
        
        # Variation 4: Brand name and dosage rephrasing
        SampleVariation(
            original_sample=samples[3],
            variation_text="Terapi med paracetamol 500 milligram, tre doser per døgn over fem dager.",
            variation_type="brand_and_phrasing",
            variation_id="no_var_004",
            changes_applied=[
                "Replaced 'behandling' with 'terapi'",
                "Replaced brand 'Paracet' with generic 'paracetamol'",
                "Replaced 'mg' with 'milligram'",
                "Replaced 'tre ganger daglig' with 'tre doser per døgn'",
                "Replaced 'i' with 'over'"
            ],
            metadata={"difficulty": "easy"}
        ),
        
        # Variation 5: Laboratory terminology variation
        SampleVariation(
            original_sample=samples[4],
            variation_text="Blodanalysen avdekker høye nivåer av kolesterol og fett i blodet.",
            variation_type="terminology_variation",
            variation_id="no_var_005",
            changes_applied=[
                "Replaced 'blodprøven' with 'blodanalysen'",
                "Replaced 'viser' with 'avdekker'",
                "Replaced 'forhøyede verdier' with 'høye nivåer'",
                "Replaced 'triglyserider' with 'fett i blodet'"
            ],
            metadata={"difficulty": "medium"}
        ),
        
        # Variation 6: Brand name and temporal expression
        SampleVariation(
            original_sample=samples[5],
            variation_text="Personen har overfølsomhet overfor pollen og behandles med cetirizin under pollenperioden.",
            variation_type="brand_and_terminology",
            variation_id="no_var_006",
            changes_applied=[
                "Replaced 'pasienten' with 'personen'",
                "Replaced 'allergi mot' with 'overfølsomhet overfor'",
                "Replaced 'får' with 'behandles med'",
                "Replaced brand 'Zyrtec' with generic 'cetirizin'",
                "Replaced 'i pollensesongen' with 'under pollenperioden'"
            ],
            metadata={"difficulty": "medium"}
        ),
    ]
    
    return variations


def get_sample_pairs_norwegian():
    """Get Norwegian medical sample pairs ready for evaluation.
    
    Returns:
        Tuple of (samples, variations) lists
    """
    samples = get_norwegian_medical_samples()
    variations = get_norwegian_medical_variations()
    return samples, variations


if __name__ == "__main__":
    """Demo: Display all Norwegian medical samples and variations."""
    print("=" * 80)
    print("Norwegian Medical Text Examples")
    print("=" * 80)
    print()
    
    samples = get_norwegian_medical_samples()
    variations = get_norwegian_medical_variations()
    
    for i, (sample, variation) in enumerate(zip(samples, variations), 1):
        print(f"Example {i}:")
        print(f"  Domain: {sample.metadata.get('domain', 'N/A')}")
        print(f"  Type: {sample.metadata.get('type', 'N/A')}")
        print(f"  Original: {sample.text}")
        print(f"  Variation: {variation.variation_text}")
        print(f"  Variation Type: {variation.variation_type}")
        print(f"  Changes: {', '.join(variation.changes_applied[:2])}...")
        print()
    
    print("=" * 80)
    print(f"Total: {len(samples)} samples with {len(variations)} variations")
    print("=" * 80)
