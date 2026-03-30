#!/usr/bin/env python3
"""Translation fine-tuning with Hugging Face (synthetic, no saving).

Uses a tiny synthetic parallel corpus and fine-tunes a small seq2seq model.
Includes a quick inference example after training.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

MODEL_NAME = "t5-small"


def build_synthetic():
    data = {
        "src": [
            "hello world",
            "good morning",
            "good night",
            "thank you",
        ],
        "tgt": [
            "hola mundo",
            "buenos dias",
            "buenas noches",
            "gracias",
        ],
    }
    return Dataset.from_dict(data)


def preprocess(batch, tokenizer):
    model_inputs = tokenizer(batch["src"], padding=True, truncation=True, max_length=32)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["tgt"], padding=True, truncation=True, max_length=32)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    ds = build_synthetic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    ds = ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    ds = ds.remove_columns(["src", "tgt"]).with_format("torch")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = TrainingArguments(
        output_dir="/tmp/hf_translation",
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
    text = "hello world"
    inputs = tokenizer(text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=10)
    print("Input:", text)
    print("Output:", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
