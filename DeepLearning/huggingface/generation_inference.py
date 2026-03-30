#!/usr/bin/env python3
"""Text generation with a pretrained GPT-2 using Hugging Face pipeline."""

from transformers import pipeline


def main():
    gen = pipeline("text-generation", model="gpt2")
    prompt = "Deep learning has changed"
    out = gen(prompt, max_new_tokens=40, do_sample=True, top_k=50, temperature=0.8)
    print(out[0]["generated_text"])


if __name__ == "__main__":
    main()
