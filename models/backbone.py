import torch
import torch.nn as nn

class MetaBERT3D(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        point_dim=3,
        hidden_size=768,
        num_layers=12,
        num_heads=8
    ):
        super().__init__()
        
        self.point_encoder = PointTransformerEncoder(
            input_channels=point_dim,
            output_channels=hidden_size,
            num_heads=num_heads
        )
        
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        self.cross_modal_attention = CrossModalAttention(
            point_dim=hidden_size,
            text_dim=hidden_size,
            joint_dim=hidden_size
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
    def forward(self, points, input_ids, attention_mask=None):
        # Encode point cloud
        point_features = self.point_encoder(points)
        
        # Encode text
        text_features = self.language_encoder(
            input_ids,
            attention_mask
        )
        
        # Cross-modal attention
        joint_features = self.cross_modal_attention(
            point_features,
            text_features
        )
        
        # Output predictions
        outputs = self.output_head(joint_features)
        return outputs
