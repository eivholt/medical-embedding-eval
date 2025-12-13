# Medical Embedding Evaluation Framework

A Python framework for evaluating and benchmarking embedding models on medical records, with a focus on Norwegian medical text. The framework evaluates embeddings by comparing semantic similarity between original samples and their variations using cosine similarity.

## Overview

This framework allows you to:
- Evaluate embedding models on medical text samples
- Compare semantic variations that preserve meaning while changing phrasing, synonyms, medical brands, procedures, and diagnoses
- Benchmark multiple embedding models against each other
- Generate metrics and reports for model performance

## Key Features

- **Flexible Sample Management**: Create medical samples with metadata and variations
- **Pluggable Embedding Models**: Easy-to-implement interface for any embedding model
- **Semantic Variation Support**: Track different types of variations (paraphrasing, synonyms, medical terms, brands)
- **Comprehensive Metrics**: Cosine similarity, aggregated statistics, and type-specific analysis
- **Norwegian Medical Examples**: Pre-built examples for Norwegian medical text evaluation
- **Batch Processing**: Efficient evaluation of multiple samples

## Installation

### From source

```bash
git clone https://github.com/eivholt/medical-embedding-eval.git
cd medical-embedding-eval
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

### Development dependencies

```bash
pip install -r requirements-dev.txt
```

## Quick Start

```python
from medical_embedding_eval import (
    MedicalSample,
    SampleVariation,
    SamplePair,
    EmbeddingEvaluator,
    DummyEmbedder,
)

# Create a medical sample
sample = MedicalSample(
    text="Pasienten har høyt blodtrykk og diabetes type 2.",
    sample_id="sample_001",
    metadata={"language": "norwegian", "domain": "cardiology"}
)

# Create a semantic variation
variation = SampleVariation(
    original_sample=sample,
    variation_text="Personen lider av forhøyet blodtrykksmåling og sukkersyke type 2.",
    variation_type="paraphrase",
    variation_id="var_001",
    changes_applied=["Replaced 'pasienten' with 'personen'", ...]
)

# Create a sample pair
pair = SamplePair(original=sample, variation=variation)

# Initialize embedding model (use your own model here)
embedder = DummyEmbedder(embedding_dim=384)

# Evaluate
evaluator = EmbeddingEvaluator(embedder)
metrics = evaluator.evaluate_pairs([pair])

# View results
print(metrics)
```

## Usage Examples

### Basic Usage

See `examples/basic_usage.py` for a complete working example:

```bash
cd examples
python basic_usage.py
```

### Norwegian Medical Examples

The framework includes pre-built Norwegian medical examples:

```python
from examples.norwegian_medical_examples import (
    get_norwegian_medical_samples,
    get_norwegian_medical_variations
)

samples = get_norwegian_medical_samples()
variations = get_norwegian_medical_variations()
```

See `examples/norwegian_medical_examples.py` for details.

## Embedding Workflow

1. Configure Azure OpenAI credentials by copying `.env.example` to `.env` and filling in the endpoint, API key, and deployment names.
2. Generate and cache embeddings locally:

```bash
python generate_embeddings.py
```

This script stores embeddings per deployment under `data/embeddings/`, along with the original text and a hash so changes can be detected automatically.

3. Evaluate cached embeddings without re-querying the API:

```bash
python evaluate_norwegian_samples.py
```

If any sample text changes, rerun the generation step to refresh the cache before re-evaluating.

## Framework Components

### 1. Medical Samples (`MedicalSample`)

Represents original medical text with metadata:

```python
sample = MedicalSample(
    text="Pasienten har høyt blodtrykk",
    sample_id="sample_001",
    metadata={"language": "norwegian", "domain": "cardiology"}
)
```

### 2. Sample Variations (`SampleVariation`)

Represents semantic variations that preserve meaning:

```python
variation = SampleVariation(
    original_sample=sample,
    variation_text="Personen lider av forhøyet blodtrykksmåling",
    variation_type="paraphrase",
    variation_id="var_001",
    changes_applied=["List of changes made"]
)
```

### 3. Embedding Models (`EmbeddingModel`)

Abstract interface for embedding models. Implement this to use your own model:

```python
from medical_embedding_eval import EmbeddingModel
import numpy as np

class MyEmbedder(EmbeddingModel):
    def embed(self, texts):
        # Your embedding logic here
        return embeddings
    
    def get_embedding_dimension(self):
        return 768
```

### 4. Evaluator (`EmbeddingEvaluator`)

Orchestrates the evaluation process:

```python
evaluator = EmbeddingEvaluator(embedding_model)

# Evaluate pairs
metrics = evaluator.evaluate_pairs(pairs)

# Batch evaluate texts
similarities = evaluator.batch_evaluate(originals, variations)

# Compare models
comparison = evaluator1.compare_models(evaluator2, pairs)
```

### 5. Metrics (`SimilarityMetrics`, `EvaluationMetrics`)

Compute and aggregate similarity metrics:

- Cosine similarity
- Mean, median, standard deviation
- Min/max values
- Metrics grouped by variation type

## Creating Your Own Variations

### Manual Creation

```python
from medical_embedding_eval.variation_generator import ManualVariationGenerator

variation = ManualVariationGenerator.create_variation(
    original_sample=sample,
    variation_text="Your varied text here",
    variation_type="paraphrase",
    changes_applied=["List of changes"]
)
```

### Using External Tools

You can generate variations using:
- Manual paraphrasing
- Large Language Models (LLMs)
- Translation and back-translation
- Medical terminology databases
- NLP tools for synonym replacement

The framework is agnostic to how variations are created—it focuses on evaluation.

## Variation Types

Common variation types you can use:
- `paraphrase`: General paraphrasing
- `synonym_replacement`: Replacing words with synonyms
- `medical_term_variation`: Changing medical terminology (e.g., "lungebetennelse" → "pneumoni")
- `brand_variation`: Changing brand names to generic names (e.g., "Paracet" → "paracetamol")
- `word_order`: Changing sentence structure
- `terminology_variation`: Using alternative medical terminology

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=medical_embedding_eval tests/
```

## Project Structure

```
medical-embedding-eval/
├── medical_embedding_eval/     # Main package
│   ├── __init__.py
│   ├── sample.py              # Data structures
│   ├── embedder.py            # Embedding model interface
│   ├── evaluator.py           # Evaluation engine
│   ├── metrics.py             # Similarity metrics
│   └── variation_generator.py # Variation utilities
├── tests/                      # Test suite
│   ├── test_sample.py
│   ├── test_evaluator.py
│   └── test_metrics.py
├── examples/                   # Usage examples
│   ├── basic_usage.py
│   └── norwegian_medical_examples.py
├── data/                       # Sample data (optional)
├── requirements.txt            # Dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Use Cases

### 1. Benchmarking Embedding Models

Compare different embedding models (Sentence-BERT, OpenAI, multilingual models) on Norwegian medical text.

### 2. Quality Assurance

Ensure embedding models maintain semantic similarity across variations of medical records.

### 3. Model Selection

Choose the best embedding model for your specific medical domain and language.

### 4. Fine-tuning Validation

Validate that fine-tuned models perform better than base models on medical text.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional Norwegian medical examples
- Integration with popular embedding models (Sentence-BERT, OpenAI, etc.)
- Automated variation generation tools
- Additional metrics and visualizations
- Support for other languages

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{medical_embedding_eval,
  author = {Eivind Holt},
  title = {Medical Embedding Evaluation Framework},
  year = {2024},
  url = {https://github.com/eivholt/medical-embedding-eval}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
