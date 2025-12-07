# Examples

This directory contains working examples demonstrating how to use the medical embedding evaluation framework.

## Available Examples

### 1. `basic_usage.py`

Basic introduction to the framework showing:
- Creating medical samples
- Creating semantic variations
- Running evaluation
- Viewing results

**Run:**
```bash
python basic_usage.py
```

### 2. `norwegian_medical_examples.py`

Pre-built Norwegian medical text examples with various types of semantic variations:
- Paraphrasing with synonyms
- Medical term variations
- Brand name changes
- Word order variations
- Terminology changes

**Run:**
```bash
python norwegian_medical_examples.py
```

**Use in your code:**
```python
from examples.norwegian_medical_examples import (
    get_norwegian_medical_samples,
    get_norwegian_medical_variations
)

samples = get_norwegian_medical_samples()
variations = get_norwegian_medical_variations()
```

### 3. `evaluate_norwegian_samples.py`

Complete end-to-end evaluation workflow:
- Load Norwegian medical samples
- Create sample pairs
- Initialize embedding model (dummy for demonstration)
- Run evaluation
- Generate comprehensive report with detailed analysis

**Run:**
```bash
python evaluate_norwegian_samples.py
```

## Quick Start

The simplest way to get started:

```bash
# Install the package
cd ..
pip install -e .

# Run basic example
cd examples
python basic_usage.py
```

## Using with Your Own Embedding Model

Replace the `DummyEmbedder` with your actual model. See the [Integration Guide](../INTEGRATION_GUIDE.md) for examples with:
- Sentence-BERT
- OpenAI Embeddings
- HuggingFace models
- Custom models

Example with Sentence-BERT:

```python
from sentence_transformers import SentenceTransformer
from medical_embedding_eval import EmbeddingModel, EmbeddingEvaluator
import numpy as np

class SentenceBERTEmbedder(EmbeddingModel):
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array(self.model.encode(texts))
    
    def get_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_name(self):
        return f"SentenceBERT({self.model_name})"

# Use it
embedder = SentenceBERTEmbedder()
evaluator = EmbeddingEvaluator(embedder)
# ... rest of evaluation
```

## Creating Your Own Examples

### Adding New Medical Samples

```python
from medical_embedding_eval import MedicalSample, SampleVariation

# Create your sample
my_sample = MedicalSample(
    text="Your Norwegian medical text here",
    sample_id="custom_001",
    metadata={"domain": "your_domain", "language": "norwegian"}
)

# Create variation
my_variation = SampleVariation(
    original_sample=my_sample,
    variation_text="Your paraphrased version",
    variation_type="paraphrase",
    variation_id="custom_var_001",
    changes_applied=["List your changes"]
)
```

### Organizing Your Data

Create a JSON file in `../data/` directory:

```json
{
  "samples": [
    {
      "sample_id": "sample_001",
      "text": "Your text",
      "metadata": {"language": "norwegian", "domain": "cardiology"}
    }
  ],
  "variations": [
    {
      "variation_id": "var_001",
      "original_id": "sample_001",
      "variation_text": "Varied text",
      "variation_type": "paraphrase",
      "changes_applied": ["change 1", "change 2"]
    }
  ]
}
```

Then load it:

```python
import json
from medical_embedding_eval import MedicalSample, SampleVariation

with open('../data/my_samples.json') as f:
    data = json.load(f)

samples = [
    MedicalSample(
        text=s['text'],
        sample_id=s['sample_id'],
        metadata=s.get('metadata', {})
    )
    for s in data['samples']
]

# Create a lookup for samples
sample_dict = {s.sample_id: s for s in samples}

variations = [
    SampleVariation(
        original_sample=sample_dict[v['original_id']],
        variation_text=v['variation_text'],
        variation_type=v['variation_type'],
        variation_id=v['variation_id'],
        changes_applied=v.get('changes_applied', [])
    )
    for v in data['variations']
]
```

## Expected Output

All examples produce console output showing:
- Number of samples evaluated
- Mean, median, and standard deviation of similarity scores
- Min/max similarity values
- Breakdown by variation type
- Individual sample results

For actual embedding models (not DummyEmbedder), you should expect:
- Cosine similarities typically between 0.5 and 0.95 for good semantic variations
- Higher similarities indicate better preservation of meaning
- Variation type analysis helps identify which types of changes are handled well

## Need Help?

- Check the main [README](../README.md) for framework overview
- See the [Integration Guide](../INTEGRATION_GUIDE.md) for model integration
- Look at the source code in `../medical_embedding_eval/` for detailed documentation
- Open an issue on GitHub for questions or bugs
