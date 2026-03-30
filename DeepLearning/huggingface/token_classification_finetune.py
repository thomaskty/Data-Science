#!/usr/bin/env python3
"""Token classification (NER-style) with Hugging Face.

Synthetic data, minimal training, no checkpoint saving.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["O", "ENT"]


def build_synthetic():
    data = {
        "tokens": [
            ["paris", "is", "in", "france"],
            ["london", "is", "a", "city"],
            ["king", "and", "queen"],
        ],
        "ner_tags": [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0],
        ],
    }
    return Dataset.from_dict(data)


def tokenize_and_align_labels(batch, tokenizer):
    tokenized = tokenizer(batch["tokens"], is_split_into_words=True, padding=True, truncation=True)
    labels = []
    for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(batch["tokens"]))):
        label_ids = []
        word_labels = batch["ner_tags"][i]
        prev = None
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev:
                label_ids.append(word_labels[w])
            else:
                label_ids.append(-100)
            prev = w
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


def main():
    ds = build_synthetic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ds = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    ds = ds.remove_columns(["tokens", "ner_tags"]).with_format("torch")

    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))

    args = TrainingArguments(
        output_dir="/tmp/hf_token_cls",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)
    trainer.train()

    # Quick test
    sample = ["paris", "is", "a", "city"]
    inputs = tokenizer(sample, is_split_into_words=True, return_tensors="pt")
    logits = model(**inputs).logits
    preds = np.argmax(logits.detach().numpy(), axis=-1)[0]
    print("Tokens:", sample)
    print("Preds:", [LABELS[p] for p in preds])


if __name__ == "__main__":
    main()
