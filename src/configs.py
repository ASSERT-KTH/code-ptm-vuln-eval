import os
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

MODEL_CONFIGS = {
    "codebert": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/codebert-base",
        "max_seq_length": 256,
    },
}

TASK_CONFIGS = {
    "vul": {
        "base_dir": "data/vul/primevul",
        "train_file": "train.parquet",
        "test_file": "test.parquet",
        "valid_file": "valid.parquet",
    },
}

def get_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def get_task(task_name):
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Task '{task_name}' is not supported. Available tasks: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[task_name]