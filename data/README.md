# Data Directory

This directory can be used to store medical sample data for evaluation.

## Structure

You can organize your data in various formats:

### JSON Format

```json
{
  "samples": [
    {
      "sample_id": "sample_001",
      "text": "Pasienten har høyt blodtrykk",
      "metadata": {
        "language": "norwegian",
        "domain": "cardiology"
      }
    }
  ],
  "variations": [
    {
      "variation_id": "var_001",
      "original_id": "sample_001",
      "variation_text": "Personen lider av forhøyet blodtrykksmåling",
      "variation_type": "paraphrase",
      "changes_applied": ["Replaced 'pasienten' with 'personen'"]
    }
  ]
}
```

### CSV Format

You can also use CSV files with columns:
- `sample_id`
- `text`
- `language`
- `domain`

## Loading Data

You can load your data and convert it to framework objects:

```python
import json
from medical_embedding_eval import MedicalSample, SampleVariation

# Load from JSON
with open('data/my_samples.json') as f:
    data = json.load(f)

samples = [
    MedicalSample(
        text=s['text'],
        sample_id=s['sample_id'],
        metadata=s.get('metadata', {})
    )
    for s in data['samples']
]
```

## Privacy and Security

⚠️ **Important**: Do not commit real patient data or any personally identifiable information (PII) to version control. This directory should only contain:
- Synthetic/anonymized data
- Example templates
- Test data

If working with real medical records, ensure compliance with:
- GDPR (EU)
- HIPAA (US)
- Local data protection regulations
