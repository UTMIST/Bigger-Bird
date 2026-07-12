import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model import BiggerBirdAttention
from biggerbirdconfigs import BiggerBirdConfig
from shared.patched_model import classification_forward


def extend_bigbird_embeddings(model, context_len: int):
    """Extend BigBird-RoBERTa position embeddings to support sequences longer than 4096."""
    emb = model.bert.embeddings
    old_pos_emb = emb.position_embeddings
    old_max, hidden_size = old_pos_emb.weight.shape

    padding_idx = old_pos_emb.padding_idx
    if padding_idx is None:
        padding_idx = 0

    new_max = context_len + padding_idx + 1

    if new_max > old_max:
        new_pos_emb = nn.Embedding(
            new_max,
            hidden_size,
            padding_idx=padding_idx,
        ).to(device=old_pos_emb.weight.device, dtype=old_pos_emb.weight.dtype)

        with torch.no_grad():
            new_pos_emb.weight[:old_max] = old_pos_emb.weight
            new_pos_emb.weight[old_max:] = old_pos_emb.weight[-1].unsqueeze(0)

        emb.position_embeddings = new_pos_emb

    emb.register_buffer(
        "position_ids",
        torch.arange(new_max, device=old_pos_emb.weight.device).expand((1, -1)),
        persistent=False,
    )

    emb.register_buffer(
        "token_type_ids",
        torch.zeros((1, new_max), dtype=torch.long, device=old_pos_emb.weight.device),
        persistent=False,
    )

    model.config.max_position_embeddings = new_max
    model.bert.config.max_position_embeddings = new_max

    return model


class PatchedModel(nn.Module):
    def __init__(self, base_model, context_len=4096, **config_kwargs):
        super().__init__()
        self.model = base_model
        self.context_len = context_len

        if context_len > base_model.config.max_position_embeddings:
            extend_bigbird_embeddings(self.model, context_len)

        bigger_config = BiggerBirdConfig(**config_kwargs)

        for layer in self.model.bert.encoder.layer:
            old_attn = layer.attention.self
            new_attn = BiggerBirdAttention(bigger_config)
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            layer.attention.self = new_attn

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return classification_forward(self.model, input_ids, attention_mask, labels, **kwargs)

    @property
    def config(self):
        return self.model.config
