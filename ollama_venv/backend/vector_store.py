"""
Handles FAISS vector store creation and loading.
"""

import faiss
import numpy as np
import os
from backend.config import INDEX_FILE


def load_or_create_index(embeddings):
    """
    Loads existing FAISS index or creates new one.
    """

    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    return index