"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import random

class FeatureMemory:
    def  __init__(self,  memory_per_class=2048, feature_size=256):
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.memory = {}
        # self.n_classes = n_classes


    def add_features_from_sample_learned(self,  features, class_labels):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()
        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            # 这就是一个二维矩阵
            features_c = features[mask_c, :] # get features from class c
            if features_c.shape[0] > 0:
                if self.memory[c] is None:
                     self.memory[c] = features.cpu().numpy()[:self.memory_per_class, :]
                else:
                    with torch.no_grad():
                        memory_c = torch.from_numpy(self.memory[c]).cuda()
                        corr = torch.mm(features_c, memory_c.transpose(0, 1))
                        related_bank_corr, related_bank_idx = corr.max(dim= 1)
                        # # 对应feature中的索引
                        # ema_idx = (related_bank_corr >= 0.95).nonzero(as_tuple=False)
                        add_idx = (related_bank_corr < 0.95).nonzero(as_tuple=False)
                        # feature_ema = features_c [ema_idx.T[0]]
                        # class_related_bank_idx = related_bank_idx[ema_idx.T[0]].view(-1, 1)
                        # # 根据特征编号计算每个编号对应的平均特征
                        # unique_indices, inverse_indices = torch.unique(class_related_bank_idx, return_inverse=True)
                        # new_features = torch.zeros(len(unique_indices), feature_ema.size(1)).cuda()
                        # new_features.scatter_add_(0, inverse_indices.expand(-1, feature_ema.size(1)), feature_ema)
                        # counts = torch.bincount(inverse_indices.T[0])
                        # new_features /= counts.unsqueeze(1)
                        # # 保持原有的编号
                        # new_feature_indices = unique_indices.unsqueeze(1)
                        # memory_c[new_feature_indices.T[0]] = 0.99 * memory_c[new_feature_indices.T[0]] + 0.01 * new_features
                        memory_c = torch.cat([features_c[add_idx.T[0]], memory_c], dim=0)
                        memory_c = memory_c.cpu().numpy()[:self.memory_per_class, :]
                        self.memory[c] = memory_c