"""
RAG Query Pipeline:
1. Embed query
2. Search FAISS
3. Build context
4. Generate LLM answer
"""

import numpy as np
import time
from ollama import chat
from backend.config import LLM_MODEL
from backend.embeddings import get_embedding


def query_rag(index, docs, query, cache, stream_callback=None):
    """
    Main RAG pipeline.
    """

    start_total = time.time()

    # Embed Query
    query_vector = get_embedding(query, cache)

    # FAISS Search
    D, I = index.search(
        np.array([query_vector]).astype("float32"), k=2
    )

    retrieved_docs = [docs[i] for i in I[0]]

    # Build Context
    context = ""
    for doc in retrieved_docs:
        context += f"\nSource: {doc.metadata.get('source')}\n"
        context += doc.page_content + "\n"

    system_prompt = """
    You are a strict academic auditor.
    Answer ONLY from the provided context.
    If the answer is not found, say:
    'Information not available in documents.'
    Do NOT guess.
    Keep answers concise and under 100 words.
    """

    # LLM Call
    response_text = ""

    for chunk in chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        stream=True
    ):
        token = chunk["message"]["content"]
        response_text += token

        if stream_callback:
            stream_callback(response_text)

    return response_text, retrieved_docs