# Code PTM Vulnerability Detection Evaluation Framework

This repository contains the evaluation framework for assessing Code Pre-trained Models (Code PTMs) on vulnerability detection task, developed as part of a master's thesis research.

## Overview

The framework provides a comprehensive pipeline for evaluating various encoder-only and encoder-decoder Code PTMs on C/C++ vulnerability detection datasets (PrimeVul, DiverseVul and Devign). The evaluation process follows a four-stage approach: data processing, feature extraction, classification, and evaluation.

## Framework Architecture

The evaluation framework consists of the following components:

1. **Data Processing**: Load and preprocess function-level C/C++ code snippets with vulnerability labels
2. **Feature Extraction**: Extract fixed-size embeddings from Code PTMs using various pooling strategies
3. **Classification**: Train lightweight neural classifiers on the extracted embeddings
4. **Evaluation**: Assess performance using macro F1 score with cross-dataset evaluation support

## Supported Models

### Encoder-Only Models

- **CodeBERT** (base)
- **GraphCodeBERT** (base)
- **UniXcoder** variants (base, base-unimodal, base-nine)
- **CodeSage** variants (small-v2, base-v2, large-v2)
- **ModernBERT** (base, large)

### Encoder-Decoder Models

- **CodeT5** variants (small, base, large)
- **CodeT5+** variants (220m, 220m-bimodal, 770m)
- **CoditT5** (base)
- **AST-T5** (base)
- **DivoT5** variants (60m, 220m)

### Baseline

- **Random Embedding Model** for statistical significance testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Feature Extraction

Extract embeddings from Code PTMs for all dataset splits:

```bash
python extractor.py \
    --model codebert-base \
    --task vul \
    --dataset primevul \
    --batch_size 32 \
    --seed 42
```

> **Extracted embeddings and best classifier checkpoint are available on Zenodo:** [https://doi.org/10.5281/zenodo.19147437](https://doi.org/10.5281/zenodo.19147437)
> Download and place the feature files under `data/` and the checkpoint files under `checkpoints/`.

### 2. Train the Classifier

Train classifiers on extracted features (runs 10 seeds, saves the best checkpoint per seed):

```bash
python classifier.py \
    --task vul \
    --model codebert-base \
    --dataset primevul \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 5e-5 \
    --method CLS \
    --mode train \
    --wandb
```

### 3. Evaluate the Classifier

Load saved checkpoints and evaluate on the test set (reports per-seed results and aggregated mean ± std):

```bash
python classifier.py \
    --task vul \
    --model codebert-base \
    --dataset primevul \
    --test_dataset diversevul \
    --batch_size 16 \
    --method CLS \
    --mode eval
```

To evaluate a single checkpoint, use `--checkpoint`:

```bash
python classifier.py \
    --task vul \
    --model codebert-base \
    --dataset primevul \
    --test_dataset diversevul \
    --method CLS \
    --mode eval \
    --checkpoint checkpoints/vul_primevul_codebert-base_CLS_seed42_best.pt
```

### 4. Random Baseline Experiments

Run comprehensive random baseline evaluation:

```bash
python experiments_random.py \
    --task vul \
    --dataset primevul \
    --test_dataset diversevul \
    --n_embed_seeds 100 \
    --embed_seed_start 73 \
    --extract \
    --batch_size 16 \
    --num_epochs 10 \
    --method CLS
```

## Key Parameters

### Pooling Methods

- `CLS`: Extract the embedding from the special [CLS] token position
- `EOS`: Extract the embedding from the special <eos> (end-of-sequence) token position
- `AVG`: Compute the average of all token embeddings, excluding padding tokens
- `MAX`: Take the element-wise maximum across all token embeddings, excluding padding tokens

### Model Configuration

- `--model`: Specify which Code PTM to use
- `--task`: Task type (currently supports 'vul' for vulnerability detection)
- `--dataset`: Training/validation dataset
- `--test_dataset`: Test dataset (optional, defaults to same as training dataset)

### Training Configuration

- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for AdamW optimizer
- `--seed`: Random seed for reproducibility

## Cross-Dataset Evaluation

The framework supports cross-dataset evaluation to assess model generalization.
Train on one dataset, then evaluate on another by passing a different `--test_dataset`:

```bash
# Train on PrimeVul
python classifier.py \
    --task vul \
    --dataset primevul \
    --model codebert-base \
    --method CLS \
    --mode train

# Evaluate on DiverseVul
python classifier.py \
    --task vul \
    --dataset primevul \
    --test_dataset diversevul \
    --model codebert-base \
    --method CLS \
    --mode eval
```

## Configuration

All model and dataset configurations are centralized in `configs.py`:

- `MODEL_CONFIGS`: Model-specific parameters (tokenizer, max length, pooling options)
- `TASK_CONFIGS`: Dataset-specific file paths and configurations
