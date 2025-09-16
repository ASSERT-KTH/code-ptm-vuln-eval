import torch
from torch import nn


class RandomEmbeddingModel(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, **kwargs):
        b, L = input_ids.shape
        device = input_ids.device
        last_hidden_state = torch.randn(b, L, self.hidden_size, device=device)
        return type(
            "Obj",
            (object,),
            {
                "hidden_states": [last_hidden_state],
                "last_hidden_state": last_hidden_state,
                "encoder_last_hidden_state": last_hidden_state,
            },
        )()


class DummyConfig:
    def __init__(self, hidden_size=768):
        self.hidden_size = hidden_size

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class DummyTokenizer:
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    eos_token = "[EOS]"
    cls_token_id = 101
    eos_token_id = 102
    sep_token_id = 103

    def __init__(self):
        self.special_token_map = {
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.eos_token: self.eos_token_id,
        }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def tokenize(self, text: str):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [self.special_token_map.get(tok, 1) for tok in tokens]
