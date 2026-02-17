#!/usr/bin/env python3
"""Topic classification (synthetic) with Hugging Face.

Simple dataset with 3 topics, no checkpoint saving.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["sports", "finance", "tech"]


def build_synthetic():
    data = {
        "text": [
            "the team won the league",
            "stocks rallied after earnings",
            "new ai model released",
            "the player scored a goal",
            "the market crashed today",
            "advances in machine learning",
        ],
        "label": [0, 1, 2, 0, 1, 2],
    }
    return Dataset.from_dict(data)


def main():
    ds = build_synthetic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["text"]).with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir="/tmp/hf_topic",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()

    sample = "the company reported strong quarterly earnings"
    inputs = tokenizer(sample, return_tensors="pt")
    logits = model(**inputs).logits
    pred = logits.argmax(dim=1).item()
    print("Text:", sample)
    print("Topic:", LABELS[pred])


if __name__ == "__main__":
    main()
