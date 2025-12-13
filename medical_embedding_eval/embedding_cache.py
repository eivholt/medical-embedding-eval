"""Persistence utilities for caching embeddings on disk."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def compute_text_hash(text: str) -> str:
    """Compute a stable hash for a text payload."""
    normalized = text.replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class CachedEmbedding:
    """Container for a cached embedding entry."""

    item_id: str
    item_type: str
    text: str
    text_hash: str
    embedding: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "text": self.text,
            "text_hash": self.text_hash,
            "embedding": self.embedding.astype(float).tolist(),
        }


class EmbeddingCache:
    """Cache manager that stores embeddings per model deployment."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    @staticmethod
    def record_key(item_type: str, item_id: str) -> str:
        return f"{item_type}:{item_id}"

    def _path_for(self, model_key: str) -> Path:
        return self.cache_dir / f"{model_key}.json"

    def load(self, model_key: str) -> Dict[str, CachedEmbedding]:
        path = self._path_for(model_key)
        if not path.exists():
            return {}

        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        records: Dict[str, CachedEmbedding] = {}
        for entry in payload.get("records", []):
            key = self.record_key(entry["item_type"], entry["item_id"])
            vector = np.asarray(entry["embedding"], dtype=np.float32)
            records[key] = CachedEmbedding(
                item_id=entry["item_id"],
                item_type=entry["item_type"],
                text=entry.get("text", ""),
                text_hash=entry.get("text_hash", ""),
                embedding=vector,
            )
        return records

    def save(self, model_key: str, records: Dict[str, CachedEmbedding]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(model_key)
        payload = {
            "model": model_key,
            "records": [record.to_dict() for record in records.values()],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def update_records(
        self,
        model_key: str,
        records: Dict[str, CachedEmbedding],
        new_entries: Iterable[CachedEmbedding],
    ) -> None:
        for entry in new_entries:
            key = self.record_key(entry.item_type, entry.item_id)
            records[key] = entry
        self.save(model_key, records)
