from __future__ import annotations

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from random_model import DummyConfig, DummyTokenizer, RandomEmbeddingModel

MODEL_CONFIGS = {
    "codebert-base": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/codebert-base",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": False,
        "is_encoder_decoder": False,
    },
    "graphcodebert-base": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/graphcodebert-base",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": False,
        "is_encoder_decoder": False,
    },
    "unixcoder-base-unimodal": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/unixcoder-base-unimodal",
        "max_seq_length": 1024,
        "pool_cls": True,
        "pool_eos": False,
        "is_encoder_decoder": False,
    },
    "unixcoder-base": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/unixcoder-base",
        "max_seq_length": 1024,
        "pool_cls": True,
        "pool_eos": False,
        "is_encoder_decoder": False,
    },
    "unixcoder-base-nine": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
        "model_path": "microsoft/unixcoder-base-nine",
        "max_seq_length": 1024,
        "pool_cls": True,
        "pool_eos": False,
        "is_encoder_decoder": False,
    },
    "codesage-small-v2": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSequenceClassification,
        "model_path": "codesage/codesage-small-v2",
        "max_seq_length": 2048,
        "pool_cls": False,
        "pool_eos": True,
        "is_encoder_decoder": False,
    },
    "codesage-base-v2": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSequenceClassification,
        "model_path": "codesage/codesage-base-v2",
        "max_seq_length": 2048,
        "pool_cls": False,
        "pool_eos": True,
        "is_encoder_decoder": False,
    },
    "codesage-large-v2": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSequenceClassification,
        "model_path": "codesage/codesage-large-v2",
        "max_seq_length": 2048,
        "pool_cls": False,
        "pool_eos": True,
        "is_encoder_decoder": False,
    },
    "modernbert-base": {
        "config": ModernBertConfig,
        "tokenizer": AutoTokenizer,
        "model": ModernBertForSequenceClassification,
        "model_path": "answerdotai/ModernBERT-base",
        "max_seq_length": 8192,
        "pool_cls": True,
        "pool_eos": False,
        "is_encoder_decoder": False,
    },
    "modernbert-large": {
        "config": ModernBertConfig,
        "tokenizer": AutoTokenizer,
        "model": ModernBertForSequenceClassification,
        "model_path": "answerdotai/ModernBERT-large",
        "max_seq_length": 8192,
        "pool_cls": True,
        "pool_eos": False,
        "is_encoder_decoder": False,
    },
    "codet5-small": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "Salesforce/codet5-small",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "codet5-base": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "Salesforce/codet5-base",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "codet5-large": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "Salesforce/codet5-large",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "codet5p-220m-bimodal": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModel,
        "model_path": "Salesforce/codet5p-220m-bimodal",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "codet5p-220m": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "Salesforce/codet5p-220m",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "codet5p-770m": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "Salesforce/codet5p-770m",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "codit5": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "JiyangZhang/CoditT5",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "ast-t5": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "gonglinyuan/ast_t5_base",
        "max_seq_length": 1024,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "divot5-60m": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "qyliang/DivoT5-60M",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "divot5-220m": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSeq2SeqLM,
        "model_path": "qyliang/DivoT5-220M",
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": True,
    },
    "random-embed": {
        "config": DummyConfig,
        "tokenizer": DummyTokenizer,
        "model": RandomEmbeddingModel,
        "model_path": None,
        "max_seq_length": 512,
        "pool_cls": True,
        "pool_eos": True,
        "is_encoder_decoder": False,
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
            },
            "primevul_new": {
                "base_dir": "data/vul/primevul_new",
                "train_file": "primevul_train_paired.jsonl",
                "test_file": "primevul_test_paired.jsonl",
                "valid_file": "primevul_valid_paired.jsonl",
            },
            "diversevul": {
                "base_dir": "data/vul/diversevul",
                "train_file": "train.parquet",
                "test_file": "test.parquet",
                "valid_file": "valid.parquet",
            },
            "diversevul_balanced": {
                "base_dir": "data/vul/diversevul_balanced",
                "train_file": "train.parquet",
                "test_file": "test.parquet",
                "valid_file": "valid.parquet",
            },
        }
    }
}


def get_model(name: str) -> dict:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[name]


def get_task(task_name: str, dataset_name: str) -> dict:
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASK_CONFIGS)}")
    task = TASK_CONFIGS[task_name]
    if dataset_name not in task["datasets"]:
        raise ValueError(
            f"Unknown dataset '{dataset_name}' for task '{task_name}'. "
            f"Available: {list(task['datasets'])}"
        )
    return task["datasets"][dataset_name]
