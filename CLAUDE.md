# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

An educational ML repository with implementations of classical and modern deep learning architectures using PyTorch. Each directory contains either a Jupyter notebook (standalone learning exercise) or Python modules (more complete implementation).

## Environment Setup

- Package manager: `uv`
- Python version: 3.12
- Virtual environment: `.venv` at project root

```bash
# Create (already done)
uv venv .venv --python 3.12

# Activate
source .venv/bin/activate

# Install packages
uv pip install <package>
```

Always use `.venv` when running Python scripts or installing packages.

## Running Notebooks

Notebooks are designed for Google Colab or local Jupyter:
```bash
jupyter notebook <directory>/<notebook>.ipynb
```

## Structure by Type

**Standalone notebooks** (self-contained, single file):
- `simple_ann/ann.ipynb` — MLP on Iris dataset
- `simple_cnn/cnn.ipynb` — CNN on MNIST
- `alexnet/alexnet.ipynb` — AlexNet on CIFAR-10
- `vgg/vgg.ipynb` — VGG16 on CIFAR-10
- `resnet/resnet.ipynb` — ResNet family on CIFAR-10
- `Colab/StyleTTS2_Demo_LJSpeech.ipynb` — TTS inference demo

**Python module implementations**:
- `clip/` — Full CLIP implementation (image+text contrastive learning)
- `screen/` — Production multimodal screen classifier with LoRA/QLoRA
- `hftrainer/hftrainer.py` — HuggingFace Trainer integration example

**Reference only** (PDF papers, no code): `autoencoder/`, `bert/`, `gpt/`, `transformer/`, `unet/`, `vit/`

## CLIP Architecture

Entry point: `clip/main.py`. Components:
- `ImageEncoder.py` — ResNet-50 pretrained backbone
- `TextEncoder.py` — DistilBERT backbone
- `ProjectionHead.py` — Maps both modalities to 256-dim shared space
- `CFG.py` — All hyperparameters (batch size 32, 4 epochs, temp=1.0)
- `utils.py` — Contrastive loss and transforms
- Supports CIFAR-100 and Pokemon datasets

```bash
cd clip && python main.py
```

## Screen Model Architecture

Production-grade multimodal classifier. Entry points in `screen/training/`:
- `train.py` — Full fine-tuning via HF Trainer
- `lora_train.py` — LoRA fine-tuning (rank 16)
- `qlora_train.py` — QLoRA with 4-bit quantization

Model uses CLIP Vision (openai/clip-vit-base-patch32) + XLM-RoBERTa as frozen encoders with MLPHeader projection heads. Data stored in LMDB format; preprocessing scripts in `screen/preprocess/`.

Dependencies: `screen/requriements.txt` (transformers 4.25.1, peft, bitsandbytes, lmdb, albumentations)

## Common Patterns

- All models use `torch.nn.Module`
- CIFAR-10/100 is the default dataset for CNN notebooks; images resized to 64x64 or 110x110
- SGD + momentum (0.9) + `ReduceLROnPlateau` for CNN training; AdamW for transformer-based models
- Multi-level learning rates: projection heads (1e-3) > image encoder (1e-4) > text encoder (1e-5)
- Data augmentation via `albumentations`; HuggingFace `datasets` for text/multimodal data
