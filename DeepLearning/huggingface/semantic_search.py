#!/usr/bin/env python3
"""Semantic search with sentence-transformers.

Embeds a small corpus, then retrieves nearest neighbors for a query.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    corpus = [
        "deep learning for images",
        "transformers for language",
        "stocks and financial markets",
        "football match results",
    ]
    query = "neural networks for text"

    model = SentenceTransformer(MODEL_NAME)
    corpus_emb = model.encode(corpus, normalize_embeddings=True)
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = corpus_emb @ query_emb
    top = np.argsort(-scores)

    print("Query:", query)
    for i in top:
        print(f"{scores[i]:.3f} -> {corpus[i]}")


if __name__ == "__main__":
    main()
