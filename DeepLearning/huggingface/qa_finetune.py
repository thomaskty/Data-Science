#!/usr/bin/env python3
"""Extractive QA fine-tuning with Hugging Face on synthetic data.

No checkpoint saving; includes a quick inference example.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
import torch

MODEL_NAME = "distilbert-base-uncased"


def build_synthetic():
    data = {
        "context": [
            "paris is a city in france",
            "london is a city in england",
        ],
        "question": [
            "what is the capital",
            "what is the capital",
        ],
        "answers": [
            {"text": ["paris"], "answer_start": [0]},
            {"text": ["london"], "answer_start": [0]},
        ],
    }
    return Dataset.from_dict(data)


def prepare_features(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding=True,
        max_length=64,
        return_offsets_mapping=True,
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer_start = examples["answers"][i]["answer_start"][0]
        answer_text = examples["answers"][i]["text"][0]
        answer_end = answer_start + len(answer_text)

        # Find token start/end
        start_pos = 0
        end_pos = 0
        for idx, (s, e) in enumerate(offsets):
            if s <= answer_start < e:
                start_pos = idx
            if s < answer_end <= e:
                end_pos = idx
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")
    return tokenized


def main():
    ds = build_synthetic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ds = ds.map(lambda x: prepare_features(x, tokenizer), batched=True)
    ds = ds.remove_columns(["context", "question", "answers"]).with_format("torch")

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    args = TrainingArguments(
        output_dir="/tmp/hf_qa",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)
    trainer.train()

    # Inference example
    context = "paris is a city in france"
    question = "what is the capital"
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits, dim=-1).item()
    end = torch.argmax(outputs.end_logits, dim=-1).item()
    answer = tokenizer.decode(inputs["input_ids"][0][start : end + 1])
    print("Q:", question)
    print("A:", answer)


if __name__ == "__main__":
    main()
