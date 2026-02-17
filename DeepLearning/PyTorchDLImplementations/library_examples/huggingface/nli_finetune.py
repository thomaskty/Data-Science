#!/usr/bin/env python3
"""Natural Language Inference (entailment/contradiction/neutral).

Tiny synthetic dataset, fine-tunes a pretrained model without saving.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

MODEL_NAME = "roberta-base"
LABELS = ["entailment", "contradiction", "neutral"]


def build_synthetic():
    data = {
        "premise": [
            "A dog is running.",
            "A man is eating.",
            "A woman is reading.",
        ],
        "hypothesis": [
            "An animal is moving.",
            "A man is sleeping.",
            "A person is reading a book.",
        ],
        "label": [0, 1, 2],
    }
    return Dataset.from_dict(data)


def main():
    ds = build_synthetic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["premise", "hypothesis"]).with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir="/tmp/hf_nli",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()

    # Quick test
    sample = tokenizer("A cat is sleeping.", "An animal is asleep.", return_tensors="pt")
    logits = model(**sample).logits
    pred = logits.argmax(dim=1).item()
    print("Prediction:", LABELS[pred])


if __name__ == "__main__":
    main()
