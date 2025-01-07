import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import spearmanr

class MetricsCalculator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_loss = 0
        self.num_samples = 0
        self.predictions = []
        self.ground_truth = []
        
    def update(self, loss, predictions, ground_truth):
        self.total_loss += loss.item()
        self.num_samples += predictions.size(0)
        self.predictions.extend(predictions.cpu().numpy())
        self.ground_truth.extend(ground_truth.cpu().numpy())
        
    def compute_metrics(self):
        metrics = {}
        
        # Average loss
        metrics['loss'] = self.total_loss / self.num_samples
        
        # Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.ground_truth,
            self.predictions,
            average='weighted'
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Correlation coefficient
        correlation, _ = spearmanr(self.predictions, self.ground_truth)
        metrics['correlation'] = correlation
        
        return metrics

class Point3DMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.chamfer_distances = []
        self.normal_consistencies = []
        
    def compute_chamfer_distance(self, pred_points, gt_points):
        # Compute bi-directional chamfer distance
        dist_matrix = torch.cdist(pred_points, gt_points)
        
        min_dist_1, _ = torch.min(dist_matrix, dim=1)
        min_dist_2, _ = torch.min(dist_matrix, dim=0)
        
        chamfer_dist = torch.mean(min_dist_1) + torch.mean(min_dist_2)
        return chamfer_dist
        
    def compute_normal_consistency(self, pred_normals, gt_normals):
        # Compute normal consistency score
        consistency = torch.abs(torch.sum(pred_normals * gt_normals, dim=-1))
        return torch.mean(consistency)
        
    def update(self, pred_points, gt_points, pred_normals=None, gt_normals=None):
        chamfer_dist = self.compute_chamfer_distance(pred_points, gt_points)
        self.chamfer_distances.append(chamfer_dist.item())
        
        if pred_normals is not None and gt_normals is not None:
            normal_cons = self.compute_normal_consistency(pred_normals, gt_normals)
            self.normal_consistencies.append(normal_cons.item())
            
    def compute_metrics(self):
        metrics = {
            'chamfer_distance': np.mean(self.chamfer_distances)
        }
        
        if self.normal_consistencies:
            metrics['normal_consistency'] = np.mean(self.normal_consistencies)
            
        return metrics
