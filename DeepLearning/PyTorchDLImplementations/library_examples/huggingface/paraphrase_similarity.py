#!/usr/bin/env python3
"""Paraphrase / semantic similarity with Hugging Face (inference only).

Uses sentence-transformers to compute cosine similarity.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def main():
    model = SentenceTransformer(MODEL_NAME)
    s1 = "A man is playing a guitar."
    s2 = "A person is playing an instrument."
    s3 = "A cat is sleeping."

    e1, e2, e3 = model.encode([s1, s2, s3])
    print("sim(s1,s2):", cosine(e1, e2))
    print("sim(s1,s3):", cosine(e1, e3))


if __name__ == "__main__":
    main()
