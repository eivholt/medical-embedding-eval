"""Utilities for loading sample and variation definitions from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .sample import MedicalSample, SampleVariation


def load_samples_from_json(base_path: Path) -> Tuple[List[MedicalSample], List[SampleVariation]]:
    """Load samples and variations from a JSON definition file."""
    with base_path.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{base_path} is empty or contains invalid JSON") from exc

    samples: List[MedicalSample] = []
    variations: List[SampleVariation] = []

    for sample_payload in payload.get("samples", []):
        sample = MedicalSample(
            text=sample_payload["text"],
            sample_id=sample_payload["sample_id"],
            metadata=sample_payload.get("metadata", {}),
        )
        samples.append(sample)

        for idx, variation_payload in enumerate(sample_payload.get("variations", []), start=1):
            variation_id = variation_payload.get("variation_id") or f"{sample.sample_id}_var_{idx:02d}"
            label = variation_payload.get("label")
            variations.append(
                SampleVariation(
                    original_sample=sample,
                    variation_text=variation_payload["variation_text"],
                    variation_type=variation_payload["variation_type"],
                    variation_id=variation_id,
                    changes_applied=variation_payload.get("changes_applied", []),
                    metadata=variation_payload.get("metadata", {}),
                    similarity_label=label,
                )
            )

    return samples, variations
