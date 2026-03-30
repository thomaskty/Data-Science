#!/usr/bin/env python3
"""Continue pretraining BERT with MLM on tiny synthetic data.

No checkpoints saved (transient training only).
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

MODEL_NAME = "bert-base-uncased"


def main():
    data = {
        "text": [
            "the cat sat on the mat",
            "the dog sat on the rug",
            "a boy plays with a ball",
            "a girl reads a book",
        ]
    }
    ds = Dataset.from_dict(data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)

    train_ds = ds.map(tokenize, batched=True)
    train_ds = train_ds.remove_columns(["text"]).with_format("torch")

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir="/tmp/hf_mlm",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("MLM pretraining step complete.")


if __name__ == "__main__":
    main()
