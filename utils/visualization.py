import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import wandb
from typing import List, Optional

class Visualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_point_cloud(
        self,
        points: torch.Tensor,
        title: Optional[str] = None,
        color: Optional[str] = 'blue',
        size: Optional[float] = 0.1
    ) -> plt.Figure:
        """Plot 3D point cloud."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = points.cpu().numpy()
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            s=size
        )
        
        if title:
            ax.set_title(title)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        return fig

    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        text_tokens: List[str]
    ) -> plt.Figure:
        """Plot attention visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(attention_weights.cpu(), cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add token labels
        ax.set_xticks(np.arange(len(text_tokens)))
        ax.set_xticklabels(text_tokens, rotation=45, ha='right')
        
        ax.set_title('Attention Weights Visualization')
        
        return fig

    def log_training_metrics(
        self,
        metrics: dict,
        step: int,
        prefix: str = 'train'
    ):
        """Log metrics to wandb."""
        log_dict = {f'{prefix}/{k}': v for k, v in metrics.items()}
        wandb.log(log_dict, step=step)
        
    def visualize_batch(
        self,
        batch: dict,
        predictions: torch.Tensor,
        step: int
    ):
        """Visualize a batch of data with predictions."""
        points = batch['points'][0]  # Take first example in batch
        text = batch['text'][0]
        
        # Plot point cloud
        point_cloud_fig = self.plot_point_cloud(
            points,
            title=f'Point Cloud: {text}'
        )
        
        # Log to wandb
        wandb.log({
            'point_cloud': wandb.Image(point_cloud_fig),
            'text': text,
            'step': step
        })
        
        plt.close('all')
