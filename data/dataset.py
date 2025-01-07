import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from transformers import BertTokenizer
import h5py
import json

class Point3DLanguageDataset(Dataset):
    def __init__(
        self,
        point_cloud_path,
        text_annotations_path,
        split='train',
        num_points=2048,
        tokenizer_name='bert-base-uncased'
    ):
        self.num_points = num_points
        self.split = split
        
        # Load point cloud data
        with h5py.File(point_cloud_path, 'r') as f:
            self.point_clouds = f[split][:]
            
        # Load text annotations
        with open(text_annotations_path, 'r') as f:
            self.annotations = json.load(f)[split]
            
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
    def __len__(self):
        return len(self.point_clouds)
        
    def process_point_cloud(self, points):
        # Randomly sample points if necessary
        if points.shape[0] > self.num_points:
            indices = np.random.choice(
                points.shape[0],
                self.num_points,
                replace=False
            )
            points = points[indices]
        elif points.shape[0] < self.num_points:
            indices = np.random.choice(
                points.shape[0],
                self.num_points,
                replace=True
            )
            points = points[indices]
            
        # Normalize to unit sphere
        center = np.mean(points, axis=0)
        points = points - center
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / dist
        
        return torch.FloatTensor(points)
        
    def __getitem__(self, idx):
        # Get point cloud
        point_cloud = self.point_clouds[idx]
        points = self.process_point_cloud(point_cloud)
        
        # Get text annotation
        text = self.annotations[idx]['description']
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'points': points,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'text': text,
            'idx': idx
