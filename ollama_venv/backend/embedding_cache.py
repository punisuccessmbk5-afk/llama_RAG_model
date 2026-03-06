"""
Handles saving and loading embedding cache.
Prevents recomputing embeddings.
"""

import pickle
import os
from backend.config import CACHE_FILE


def load_cached_embeddings():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cached_embeddings(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)