import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12):
        super().__init__()
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=12,
            intermediate_size=3072
        )
        self.bert = BertModel(self.config)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state

class CrossModalAttention(nn.Module):
    def __init__(self, point_dim=256, text_dim=768, joint_dim=512):
        super().__init__()
        self.point_proj = nn.Linear(point_dim, joint_dim)
        self.text_proj = nn.Linear(text_dim, joint_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, point_features, text_features):
        point_proj = self.point_proj(point_features)
        text_proj = self.text_proj(text_features)
        
        attn_output, _ = self.attention(
            point_proj,
            text_proj,
            text_proj
        )
        return attn_output
