import numpy as np
import torch
from typing import Optional, Tuple
import random

class PointCloudAugmentor:
    def __init__(self, config):
        self.jitter_sigma = config.augmentation['jitter_sigma']
        self.jitter_clip = config.augmentation['jitter_clip']
        self.random_rotation = config.augmentation['random_rotation']
        self.random_scale = config.augmentation['random_scale']
        self.random_translation = config.augmentation['random_translation']

    def random_rotate_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random 3D rotation."""
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return torch.matmul(points, rotation_matrix)

    def jitter_points(self, points: torch.Tensor) -> torch.Tensor:
        """Add random jitter to point coordinates."""
        noise = np.clip(
            np.random.normal(0, self.jitter_sigma, points.shape),
            -self.jitter_clip,
            self.jitter_clip
        )
        return points + torch.from_numpy(noise).float()

    def random_scale_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        scale = np.random.uniform(0.8, 1.2)
        return points * scale

    def random_translate_points(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random translation."""
        translation = np.random.uniform(-0.2, 0.2, size=(1, 3))
        return points + torch.from_numpy(translation).float()

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if self.random_rotation:
            points = self.random_rotate_points(points)
        
        if self.random_scale:
            points = self.random_scale_points(points)
            
        if self.random_translation:
            points = self.random_translate_points(points)
            
        points = self.jitter_points(points)
        return points
