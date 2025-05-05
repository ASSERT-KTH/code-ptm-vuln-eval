import os
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, ModernBertForSequenceClassification, ModernBertConfig, AutoTokenizer, MegatronBertForSequenceClassification, MegatronBertConfig, AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModel

MODEL_CONFIGS = {
    "codebert": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/codebert-base",
        "max_seq_length": 512,
    },
    "modernbert": {
        "config": ModernBertConfig,
        "tokenizer": AutoTokenizer,
        "model": ModernBertForSequenceClassification,
        "model_path": "answerdotai/ModernBERT-base",
        "max_seq_length": 8192,
    },
    "graphcodebert": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/graphcodebert-base",
        "max_seq_length": 512,
    },
    "codesage": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSequenceClassification,
        "model_path": "codesage/codesage-base",
        "max_seq_length": 2048,
    },
    "unixcoder": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/unixcoder-base-nine",
        "max_seq_length": 1024,
    },
    "codet5": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "Salesforce/codet5-base",
        "max_seq_length": 512,
    },
    "codet5+": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "Salesforce/codet5p-220m",
        "max_seq_length": 512,
    },
}

TASK_CONFIGS = {
    "vul": {
        "datasets": {
                "devign": {
                    "base_dir": "data/vul/devign",
                    "train_file": "train.parquet",
                    "test_file": "test.parquet",
                    "valid_file": "valid.parquet",
                },
                "primevul": {
                    "base_dir": "data/vul/primevul",
                    "train_file": "train.parquet",
                    "test_file": "test.parquet",
                    "valid_file": "valid.parquet",
                }
        }
    }
}

def get_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def get_task(task_name, dataset_name):
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Task '{task_name}' is not supported. Available tasks: {list(TASK_CONFIGS.keys())}")
    
    task_config = TASK_CONFIGS[task_name]
    
    if dataset_name not in task_config["datasets"]:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported for task '{task_name}'. "
            f"Available datasets: {list(task_config['datasets'].keys())}"
        )
    
    return task_config["datasets"][dataset_name]

