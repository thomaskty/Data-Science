#!/usr/bin/env python3
"""Hugging Face fine-tuning: DistilBERT for text classification (synthetic).

- Minimal synthetic dataset
- No checkpoint saving (keeps disk clean)
- Includes quick evaluation on a tiny test set
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

MODEL_NAME = "distilbert-base-uncased"


def build_synthetic():
    train = {
        "text": [
            "this movie is great",
            "i love this film",
            "this was amazing",
            "terrible movie",
            "i hate this film",
            "this was awful",
        ],
        "label": [1, 1, 1, 0, 0, 0],
    }
    test = {
        "text": [
            "what a great movie",
            "this film is awful",
        ],
        "label": [1, 0],
    }
    return Dataset.from_dict(train), Dataset.from_dict(test)


def main():
    train_ds, test_ds = build_synthetic()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"]).with_format("torch")
    test_ds = test_ds.map(tokenize, batched=True).remove_columns(["text"]).with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir="/tmp/hf_text_cls",  # transient
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)


if __name__ == "__main__":
    main()
