# Integration Guide: Using Real Embedding Models

This guide shows how to integrate popular embedding models with the medical embedding evaluation framework.

## Table of Contents
1. [Sentence-BERT (Sentence Transformers)](#sentence-bert)
2. [OpenAI Embeddings](#openai-embeddings)
3. [Custom Models](#custom-models)

---

## Sentence-BERT

[Sentence-BERT](https://www.sbert.net/) provides state-of-the-art sentence embeddings for many languages including Norwegian.

### Installation

```bash
pip install sentence-transformers
```

### Implementation

```python
from sentence_transformers import SentenceTransformer
from medical_embedding_eval import EmbeddingModel
import numpy as np

class SentenceBERTEmbedder(EmbeddingModel):
    """Wrapper for Sentence-BERT models."""
    
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize with a Sentence-BERT model.
        
        Recommended Norwegian models:
        - "paraphrase-multilingual-MiniLM-L12-v2" (50+ languages)
        - "paraphrase-multilingual-mpnet-base-v2" (50+ languages, better quality)
        - "NbAiLab/nb-sbert-base" (Norwegian-specific)
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed(self, texts):
        """Generate embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts)
        return np.array(embeddings)
    
    def get_embedding_dimension(self):
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_name(self):
        """Get model name."""
        return f"SentenceBERT({self.model_name})"


# Usage
embedder = SentenceBERTEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
from medical_embedding_eval import EmbeddingEvaluator
evaluator = EmbeddingEvaluator(embedder)
```

---

## OpenAI Embeddings

OpenAI provides powerful embedding models through their API.

### Installation

```bash
pip install openai
```

### Implementation

```python
from openai import OpenAI
from medical_embedding_eval import EmbeddingModel
import numpy as np

class OpenAIEmbedder(EmbeddingModel):
    """Wrapper for OpenAI embedding models."""
    
    def __init__(self, model="text-embedding-3-small", api_key=None):
        """Initialize with OpenAI API.
        
        Models:
        - "text-embedding-3-small" (1536 dimensions, cost-effective)
        - "text-embedding-3-large" (3072 dimensions, better quality)
        - "text-embedding-ada-002" (1536 dimensions, previous generation)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model
    
    def embed(self, texts):
        """Generate embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def get_embedding_dimension(self):
        """Get embedding dimension."""
        if "3-small" in self.model_name:
            return 1536
        elif "3-large" in self.model_name:
            return 3072
        else:  # ada-002
            return 1536
    
    def get_model_name(self):
        """Get model name."""
        return f"OpenAI({self.model_name})"


# Usage
import os
embedder = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
from medical_embedding_eval import EmbeddingEvaluator
evaluator = EmbeddingEvaluator(embedder)
```

---

## Custom Models

You can integrate any custom model by implementing the `EmbeddingModel` interface.

### Basic Template

```python
from medical_embedding_eval import EmbeddingModel
import numpy as np

class MyCustomEmbedder(EmbeddingModel):
    """Custom embedding model implementation."""
    
    def __init__(self, config):
        """Initialize your model."""
        self.model = self._load_model(config)
    
    def _load_model(self, config):
        """Load your pre-trained model."""
        # Your model loading logic
        pass
    
    def embed(self, texts):
        """Generate embeddings.
        
        Args:
            texts: str or List[str]
            
        Returns:
            np.ndarray of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Your embedding logic
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _get_embedding(self, text):
        """Generate single embedding."""
        # Your embedding logic for a single text
        pass
    
    def get_embedding_dimension(self):
        """Return the dimension of your embeddings."""
        return 768  # Replace with your model's dimension
    
    def get_model_name(self):
        """Return a descriptive name."""
        return "MyCustomModel"
```

### Example: PyTorch Model

```python
import torch
from transformers import AutoTokenizer, AutoModel
from medical_embedding_eval import EmbeddingModel
import numpy as np

class HuggingFaceEmbedder(EmbeddingModel):
    """Wrapper for HuggingFace transformer models."""
    
    def __init__(self, model_name="NbAiLab/nb-bert-base"):
        """Initialize with a HuggingFace model.
        
        Norwegian models:
        - "NbAiLab/nb-bert-base"
        - "NbAiLab/nb-bert-large"
        - "ltg/norbert"
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def embed(self, texts):
        """Generate embeddings using mean pooling."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            
        # Mean pooling
        attention_mask = encoded['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        return embeddings
    
    def get_embedding_dimension(self):
        """Get embedding dimension."""
        return self.model.config.hidden_size
    
    def get_model_name(self):
        """Get model name."""
        return f"HuggingFace({self.model_name})"


# Usage
embedder = HuggingFaceEmbedder("NbAiLab/nb-bert-base")
from medical_embedding_eval import EmbeddingEvaluator
evaluator = EmbeddingEvaluator(embedder)
```

---

## Complete Example: Comparing Multiple Models

```python
from medical_embedding_eval import EmbeddingEvaluator, SamplePair
from examples.norwegian_medical_examples import (
    get_norwegian_medical_variations
)

# Initialize multiple models
embedder1 = SentenceBERTEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
embedder2 = HuggingFaceEmbedder("NbAiLab/nb-bert-base")
embedder3 = OpenAIEmbedder("text-embedding-3-small")

# Create evaluators
evaluator1 = EmbeddingEvaluator(embedder1)
evaluator2 = EmbeddingEvaluator(embedder2)
evaluator3 = EmbeddingEvaluator(embedder3)

# Load test data
variations = get_norwegian_medical_variations()
pairs = [
    SamplePair(original=v.original_sample, variation=v)
    for v in variations
]

# Evaluate each model
print("Evaluating models...")
metrics1 = evaluator1.evaluate_pairs(pairs)
metrics2 = evaluator2.evaluate_pairs(pairs)
metrics3 = evaluator3.evaluate_pairs(pairs)

# Compare results
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print(f"\n{metrics1.model_name}:")
print(f"  Mean Similarity: {metrics1.mean_similarity:.4f}")
print(f"\n{metrics2.model_name}:")
print(f"  Mean Similarity: {metrics2.mean_similarity:.4f}")
print(f"\n{metrics3.model_name}:")
print(f"  Mean Similarity: {metrics3.mean_similarity:.4f}")

# Direct comparison
comparison = evaluator1.compare_models(evaluator2, pairs)
print(f"\nDifference (Model 1 - Model 2): {comparison['difference']:.4f}")
```

---

## Tips for Model Selection

1. **Multilingual Models**: For Norwegian, use multilingual models trained on multiple languages
2. **Domain-Specific**: Consider fine-tuning on medical text if available
3. **Performance vs Cost**: Balance between API costs (OpenAI) and local compute (HuggingFace)
4. **Embedding Dimension**: Higher dimensions generally capture more nuance but require more resources
5. **Evaluation**: Always evaluate on your specific Norwegian medical text before deployment

## Further Reading

- [Sentence-BERT Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Norwegian NLP Resources](https://github.com/web64/norwegian-nlp-resources)
