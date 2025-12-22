"""Utilities for loading sample and variation definitions from disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .sample import MedicalSample, SampleVariation


def _humanize_dataset_name(stem: str) -> str:
    parts = re.split(r"[_\-]+", stem)
    tokens = []
    for part in parts:
        if not part:
            continue
        if part.isupper() or any(char.isdigit() for char in part):
            tokens.append(part.upper())
        elif part.islower():
            tokens.append(part.capitalize())
        else:
            tokens.append(part)
    return " ".join(tokens) if tokens else stem


def load_samples_from_json(
    base_path: Path,
    *,
    dataset_id: Optional[str] = None,
    dataset_display_name: Optional[str] = None,
) -> Tuple[List[MedicalSample], List[SampleVariation]]:
    """Load samples and variations from a JSON definition file."""
    with base_path.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{base_path} is empty or contains invalid JSON") from exc

    samples: List[MedicalSample] = []
    variations: List[SampleVariation] = []

    for sample_payload in payload.get("samples", []):
        sample_metadata = dict(sample_payload.get("metadata", {}))
        if dataset_id:
            sample_metadata.setdefault("dataset_id", dataset_id)
        if dataset_display_name:
            sample_metadata.setdefault("dataset_display_name", dataset_display_name)

        sample = MedicalSample(
            text=sample_payload["text"],
            sample_id=sample_payload["sample_id"],
            metadata=sample_metadata,
        )
        samples.append(sample)

        for idx, variation_payload in enumerate(sample_payload.get("variations", []), start=1):
            variation_id = variation_payload.get("variation_id") or f"{sample.sample_id}_var_{idx:02d}"
            label = variation_payload.get("label")
            variation_metadata = dict(variation_payload.get("metadata", {}))
            if dataset_id:
                variation_metadata.setdefault("dataset_id", dataset_id)
            if dataset_display_name:
                variation_metadata.setdefault("dataset_display_name", dataset_display_name)
            variations.append(
                SampleVariation(
                    original_sample=sample,
                    variation_text=variation_payload["variation_text"],
                    variation_type=variation_payload["variation_type"],
                    variation_id=variation_id,
                    changes_applied=variation_payload.get("changes_applied", []),
                    metadata=variation_metadata,
                    similarity_label=label,
                )
            )

    return samples, variations


def load_samples_from_directory(directory: Path) -> Tuple[List[MedicalSample], List[SampleVariation]]:
    """Load samples and variations from every JSON file in a directory."""

    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    json_files = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    all_samples: List[MedicalSample] = []
    all_variations: List[SampleVariation] = []
    seen_sample_ids: Set[str] = set()
    seen_variation_ids: Set[str] = set()

    for file_path in json_files:
        dataset_id = file_path.stem
        dataset_display = _humanize_dataset_name(dataset_id)
        samples, variations = load_samples_from_json(
            file_path,
            dataset_id=dataset_id,
            dataset_display_name=dataset_display,
        )

        for sample in samples:
            if sample.sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id '{sample.sample_id}' found in {file_path}")
            seen_sample_ids.add(sample.sample_id)
            all_samples.append(sample)

        for variation in variations:
            if variation.variation_id in seen_variation_ids:
                raise ValueError(f"Duplicate variation_id '{variation.variation_id}' found in {file_path}")
            seen_variation_ids.add(variation.variation_id)
            all_variations.append(variation)

    return all_samples, all_variations
