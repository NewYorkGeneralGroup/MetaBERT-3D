import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_channels, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        # x shape: (batch_size, num_points, input_channels)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, x.size(1))
        
        return x

class PointTransformerEncoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=256, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_channels)
        self.norm2 = nn.LayerNorm(input_channels)
        self.ffn = nn.Sequential(
            nn.Linear(input_channels, input_channels * 4),
            nn.ReLU(),
            nn.Linear(input_channels * 4, input_channels)
        )
        self.final_proj = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        # x shape: (batch_size, num_points, input_channels)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        x = self.final_proj(x)
        return x
