"""
Embedding logic with caching.
"""

import numpy as np
import ollama
from backend.config import EMBED_MODEL
from backend.embedding_cache import save_cached_embeddings


def get_embedding(text, cache):
    """
    Returns embedding from cache if available,
    otherwise generates new embedding.
    """

    if text in cache:
        return cache[text]

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )

    vector = np.array(response["embedding"])
    cache[text] = vector

    save_cached_embeddings(cache)

    return vector