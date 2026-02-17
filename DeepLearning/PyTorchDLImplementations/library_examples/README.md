# Library-based Implementations (Text / Sequence Only)

This folder contains **library-first** examples for text/sequence workflows using Hugging Face.
Everything is minimal, synthetic, and meant for learning the standard patterns.

## Structure
- `huggingface/`
  - Pretraining, fine-tuning, and inference examples
  - Uses tiny synthetic datasets (no large downloads)
  - Fine-tuning does **not** save checkpoints (to avoid storage)

## Notes
- These scripts still download pretrained weights from Hugging Face
- Each script runs independently
- Modify the synthetic data or task for your own labs
