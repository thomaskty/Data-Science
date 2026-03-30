#!/usr/bin/env python3
"""Cross-encoder reranking example.

Scores query-document pairs and ranks by relevance.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    query = "best transformer model"
    docs = [
        "This paper introduces the Transformer architecture.",
        "A guide to cooking pasta.",
        "Transformers are used in NLP and CV.",
    ]

    model = CrossEncoder(MODEL_NAME)
    pairs = [[query, d] for d in docs]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
    for score, doc in ranked:
        print(f"{score:.4f} -> {doc}")


if __name__ == "__main__":
    main()
