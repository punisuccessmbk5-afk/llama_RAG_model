"""
Central configuration file.
All global settings should live here.
"""

import os

LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

os.makedirs(STORAGE_DIR, exist_ok=True)

INDEX_FILE = os.path.join(STORAGE_DIR, "faiss_index.index")
CACHE_FILE = os.path.join(STORAGE_DIR, "embedding_cache.pkl")