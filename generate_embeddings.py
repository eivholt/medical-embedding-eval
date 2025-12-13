"""Generate and persist embeddings for Norwegian medical samples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv

from medical_embedding_eval import (
    AzureOpenAIEmbedder,
    CachedEmbedding,
    DEFAULT_AZURE_EMBEDDING_CONFIGS,
    MedicalSample,
    SampleVariation,
    compute_text_hash,
    load_samples_from_directory,
    resolve_deployment_name,
)
from medical_embedding_eval.embedding_cache import EmbeddingCache

load_dotenv()

DATA_DIR = Path("data")
CACHE_DIR = Path("data/embeddings")


def collect_payloads(
    samples: Sequence[MedicalSample],
    variations: Sequence[SampleVariation],
    cache: EmbeddingCache,
    records: Dict[str, CachedEmbedding],
) -> List[Tuple[str, str, str, str]]:
    """Collect payloads that require refreshed embeddings."""
    payloads: List[Tuple[str, str, str, str]] = []

    for sample in samples:
        text_hash = compute_text_hash(sample.text)
        key = cache.record_key("sample", sample.sample_id)
        record = records.get(key)
        if record is None or record.text_hash != text_hash:
            payloads.append(("sample", sample.sample_id, sample.text, text_hash))

    for variation in variations:
        text_hash = compute_text_hash(variation.variation_text)
        key = cache.record_key("variation", variation.variation_id)
        record = records.get(key)
        if record is None or record.text_hash != text_hash:
            payloads.append(("variation", variation.variation_id, variation.variation_text, text_hash))

    return payloads


def update_cache_for_model(
    embedder: AzureOpenAIEmbedder,
    cache: EmbeddingCache,
    samples: Sequence[MedicalSample],
    variations: Sequence[SampleVariation],
) -> int:
    cache_key = embedder.get_cache_key()
    records = cache.load(cache_key)
    payloads = collect_payloads(samples, variations, cache, records)

    if not payloads:
        return 0

    texts = [payload[2] for payload in payloads]
    embeddings = embedder.embed(texts)

    new_entries: List[CachedEmbedding] = []
    for (item_type, item_id, text, text_hash), vector in zip(payloads, embeddings):
        new_entries.append(
            CachedEmbedding(
                item_id=item_id,
                item_type=item_type,
                text=text,
                text_hash=text_hash,
                embedding=vector,
            )
        )

    cache.update_records(cache_key, records, new_entries)
    return len(new_entries)


def main() -> None:
    print("=" * 80)
    print("Embedding Generation")
    print("=" * 80)
    print()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure OpenAI configuration is missing. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file."
        )

    data_dir = DATA_DIR.resolve()
    print(f"Scanning {data_dir} for JSON sample definitions...")
    samples, variations = load_samples_from_directory(data_dir)
    print(f"Loaded {len(samples)} samples and {len(variations)} variations from {data_dir}")

    cache = EmbeddingCache(CACHE_DIR)

    for config in DEFAULT_AZURE_EMBEDDING_CONFIGS:
        deployment_name = resolve_deployment_name(config)
        print()
        print(f"Processing model {config.display_name} (deployment '{deployment_name}')")

        embedder = AzureOpenAIEmbedder(
            deployment_name=deployment_name,
            embedding_dim=config.embedding_dim,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            model_name=config.display_name,
        )

        refreshed = update_cache_for_model(embedder, cache, samples, variations)
        if refreshed:
            print(f"Cached {refreshed} embeddings for {config.display_name}.")
        else:
            print(f"Cache already up to date for {config.display_name}.")

    print()
    print("Done. Cached embeddings stored under data/embeddings.")


if __name__ == "__main__":
    main()
