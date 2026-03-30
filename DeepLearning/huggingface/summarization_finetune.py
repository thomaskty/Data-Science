#!/usr/bin/env python3
"""Summarization fine-tuning with Hugging Face (synthetic, no saving).

Uses a tiny synthetic dataset and fine-tunes a small seq2seq model.
Includes a quick inference example.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

MODEL_NAME = "t5-small"


def build_synthetic():
    data = {
        "doc": [
            "The cat sat on the mat. It was a sunny day.",
            "The dog chased the ball in the park. It was very happy.",
        ],
        "summary": [
            "Cat sat on mat.",
            "Dog chased ball.",
        ],
    }
    return Dataset.from_dict(data)


def preprocess(batch, tokenizer):
    inputs = ["summarize: " + d for d in batch["doc"]]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=64)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["summary"], padding=True, truncation=True, max_length=32)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    ds = build_synthetic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    ds = ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    ds = ds.remove_columns(["doc", "summary"]).with_format("torch")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = TrainingArguments(
        output_dir="/tmp/hf_summarization",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Inference example
    text = "The cat sat on the mat. It was a sunny day."
    inputs = tokenizer("summarize: " + text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20)
    print("Input:", text)
    print("Summary:", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
