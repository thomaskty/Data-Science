#!/usr/bin/env python3
"""Inference with a pretrained Hugging Face text classifier.

Uses a pretrained DistilBERT (no fine-tune checkpoint required).
"""

from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def main():
    clf = pipeline("text-classification", model=MODEL_NAME)
    samples = [
        "This movie was surprisingly good and well-acted.",
        "I did not like the film at all.",
    ]
    for s in samples:
        print(s, "->", clf(s))


if __name__ == "__main__":
    main()
