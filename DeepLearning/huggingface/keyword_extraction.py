#!/usr/bin/env python3
"""Keyword extraction using a zero-shot classifier.

Not true extraction, but a common practical pattern: score candidate keywords.
"""

from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"


def main():
    text = "Transformers are widely used for NLP tasks like translation and summarization."
    candidates = ["transformers", "sports", "finance", "translation", "summarization"]

    clf = pipeline("zero-shot-classification", model=MODEL_NAME)
    out = clf(text, candidates, multi_label=True)

    print("Text:", text)
    for label, score in zip(out["labels"], out["scores"]):
        print(f"{label}: {score:.3f}")


if __name__ == "__main__":
    main()
