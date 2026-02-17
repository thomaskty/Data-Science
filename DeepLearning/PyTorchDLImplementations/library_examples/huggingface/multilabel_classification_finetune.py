#!/usr/bin/env python3
"""Multi-label text classification with Hugging Face (synthetic).

Uses a tiny dataset with multi-hot labels.
No checkpoint saving. Includes a quick inference example.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["sports", "finance", "politics"]


def build_synthetic():
    data = {
        "text": [
            "The team won the match and the fans cheered",
            "The stock market rallied after the earnings report",
            "The election debate focused on healthcare",
            "The government announced new economic policy",
        ],
        "labels": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
        ],
    }
    return Dataset.from_dict(data)


def main():
    ds = build_synthetic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["text"]).with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir="/tmp/hf_multilabel",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Inference example
    sample = "The election and the economy are in the news"
    inputs = tokenizer(sample, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze(0).numpy()
    print("Text:", sample)
    print("Probs:", {LABELS[i]: float(probs[i]) for i in range(len(LABELS))})


if __name__ == "__main__":
    main()
