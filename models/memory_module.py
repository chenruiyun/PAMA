import torch
import torch.nn.functional as F
import os
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors

class MemoryBank:
    def __init__(self, normal_dataset, nb_memory_sample: int = 1, device='cpu'):
        self.device = device

        # memory bank
        self.memory_information = {}
        self.yichang={}

        self.memory_per_class=2000
        # normal dataset
        self.normal_dataset = normal_dataset

        # the number of samples saved in memory bank
        self.nb_memory_sample = nb_memory_sample


    def update(self, feature_extractor):
        feature_extractor.eval()

        # define sample index
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)

        # extract features and save features into memory bank
        with torch.no_grad():
            for i in range(self.nb_memory_sample):
                # select image
                input_normal, _, _ = self.normal_dataset[samples_idx[i]]  #没有生成异常,放memory bank
                input_normal = input_normal.to(self.device)

                # extract features
                features = feature_extractor(input_normal.unsqueeze(0))

                # save features into memoery bank
                for l, features_l in enumerate(features[1:-1]):
                    if f'level{l}' not in self.memory_information.keys():
                        self.memory_information[f'level{l}'] = features_l

                    else:
                        self.memory_information[f'level{l}'] = torch.cat([self.memory_information[f'level{l}'], features_l], dim=0)
                    self.yichang[f'{l}']=None
    def update2(self, features_for_update,masks):
        # feature_extractor.eval()

        mask={}
        mask[f'mask0']=F.interpolate(masks.unsqueeze(1).float(), size=(64,64)).view(-1)
        mask[f'mask1'] = F.interpolate(masks.unsqueeze(1).float(), size=(32, 32)).view(-1)
        mask[f'mask2'] = F.interpolate(masks.unsqueeze(1).float(), size=(16, 16)).view(-1)
        # define sample index
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)

        # extract features and save features into memory bank

        with torch.no_grad():
            features=features_for_update
            for l, level in enumerate(self.memory_information.keys()):
                b, c, h, w = features[l].shape
                sample_expanded = self.memory_information[level]
                feature = features[l].detach()
                # feature=feature.permute(0,2,3,1)
                # feature[(mask[f'mask{l}']==1).detach().expand(-1, -1, -1, c)]=0
                # feature=feature.permute(0,3,1,2)

                sample_flat=sample_expanded.permute(0,2,3,1).reshape(self.memory_information[level].size(0)*h*w,-1) #1 64 64 64
                feature_flat=feature.permute(0,2,3,1).reshape(b*h*w,-1) #16 64 64 64

                similarity_matrix = torch.mm(sample_flat, feature_flat.permute(1,0))/ \
                                    (torch.norm(sample_flat, dim=1)[:, None] * torch.norm(feature_flat, dim=1))

                best_matching_indices = similarity_matrix.argmax(dim=1)
                similarities = similarity_matrix[torch.arange(h*w), best_matching_indices]
                selected_features=feature_flat[best_matching_indices]

                yichang_indices = torch.nonzero(mask[f'mask{l}'][best_matching_indices] == 1).squeeze()

                # 定义EMA更新程度
                ema_alpha = 0.90

                # 找到相似度大于0.95的索引
                similarities_threshold = 0.85
                high_similarity_indices = (similarities > similarities_threshold)
                if yichang_indices.numel() != 0:
                    high_similarity_indices[yichang_indices]=False

                update_indices = torch.arange(high_similarity_indices.size(0))[high_similarity_indices]
                sample_flat[update_indices] = ema_alpha * sample_flat[update_indices] + (
                                1 - ema_alpha) * selected_features[update_indices]
                self.memory_information[level]=sample_flat.permute(1,0).reshape(self.memory_information[level].size(0)
                                                                   ,self.memory_information[level].size(1),h,w)

                ####这里开始存储异常的memory bank
                # feature_flat
                mask_c=mask[f'mask{l}']==1
                features_c = feature_flat[mask_c, :]
                if self.yichang[f'{l}'] is None:
                    self.yichang[f'{l}'] = features_c[:self.memory_per_class, :]
                else:
                    with torch.no_grad():
                        memory_c = self.yichang[f'{l}']
                        corr = torch.mm(features_c, memory_c.transpose(0, 1))
                        related_bank_corr, related_bank_idx = corr.max(dim=1)
                        # # 对应feature中的索引
                        ema_idx = (related_bank_corr >= 0.90).nonzero(as_tuple=False)
                        add_idx = (related_bank_corr < 0.90).nonzero(as_tuple=False)
                        feature_ema = features_c [ema_idx.T[0]]
                        class_related_bank_idx = related_bank_idx[ema_idx.T[0]].view(-1, 1)
                        # # 根据特征编号计算每个编号对应的平均特征
                        unique_indices, inverse_indices = torch.unique(class_related_bank_idx, return_inverse=True)
                        new_features = torch.zeros(len(unique_indices), feature_ema.size(1)).cuda()
                        new_features.scatter_add_(0, inverse_indices.expand(-1, feature_ema.size(1)), feature_ema)
                        counts = torch.bincount(inverse_indices.T[0])
                        new_features /= counts.unsqueeze(1)
                        # # 保持原有的编号
                        new_feature_indices = unique_indices.unsqueeze(1)
                        memory_c[new_feature_indices.T[0]] = 0.9 * memory_c[new_feature_indices.T[0]] + 0.1 * new_features
                        memory_c = torch.cat([features_c[add_idx.T[0]], memory_c], dim=0)
                        memory_c = memory_c.detach()[:self.memory_per_class, :]
                        self.yichang[f'{l}'] = memory_c





    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        # batch size X the number of samples saved in memory
        # diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)
        diff_bank={}
        defect_bank={}
        # level
        #输入如果是16 64 64 64  memory是 16 64 64 64  b c h w
        with torch.no_grad():
            for l, level in enumerate(self.memory_information.keys()):

                b,c,h,w=features[l].shape
                sample_expanded= self.memory_information[level]
                feature=features[l]
                feature_flat = feature.permute(0,2,3,1).reshape(b*h*w,-1).detach()
                sample_flat = sample_expanded.permute(0,2,3,1).reshape(self.memory_information[level].size(0)*h*w,-1)
                # feature_flat=F.normalize(feature_flat,dim=1)
                # sample_flat=F.normalize(sample_flat,dim=1)
                similarity_matrix = torch.mm(feature_flat, sample_flat.permute(1,0))/\
                                    (torch.norm(feature_flat, dim=1)[:, None] * torch.norm(sample_flat, dim=1))
                best_matching_indices = similarity_matrix.argmax(dim=1)
                similarity_feature=sample_flat[best_matching_indices].reshape(b,h,w,c).permute(0,3,1,2)
                diff=F.mse_loss(input=similarity_feature,target=features[l],reduction ='none')
                diff_bank[f'{l}']=diff

                sample=self.yichang[f'{l}']
                feature_c= feature.permute(0,2,3,1).reshape(b*h*w,-1).detach()
                if sample is None:
                    defect_bank[f'{l}'] = diff
                else:
                    # defect_bank[f'{l}'] = diff
                    similarity_mat = torch.mm(feature_c, sample.permute(1, 0)) / \
                                     (torch.norm(feature_c, dim=1)[:, None] * torch.norm(sample, dim=1))
                    best_matching = similarity_mat.argmax(dim=1)
                    similarities_c = similarity_mat[torch.arange(best_matching.size(0)), best_matching]
                    similarities_threshold_c = 0.90
                    high_similarity_indices_c = (similarities_c > similarities_threshold_c)
                    update_indices_c = torch.arange(high_similarity_indices_c.size(0))[high_similarity_indices_c]
                    selected_feature_c=sample[best_matching]
                    alpha=1.0
                    # simfeature = sample[best_matching].reshape(b, h, w, c).permute(0, 3, 1, 2)
                    feature_c[update_indices_c] = alpha * selected_feature_c[update_indices_c]+(1.0-alpha)*feature_c[update_indices_c]
                    simfeature=feature_c.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    diff2 = F.mse_loss(input=simfeature, target=features[l], reduction='none')
                    defect_bank[f'{l}']=diff2
                    # print('next')

        return diff_bank,defect_bank


    def select(self, features: List[torch.Tensor]) -> torch.Tensor:
        # calculate difference between features and normal features of memory bank
        diff_bank,defect_bank = self._calc_diff(features=features)
        a = diff_bank['0'][0] - defect_bank['0'][0]
        if(a.max()==0):
            print('error!!!!!!!')
        # concatenate features with minimum difference features of memory bank
        for l, level in enumerate(self.memory_information.keys()):

            # selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))

            # diff_features = F.mse_loss(diff_bank[f'{l}'], features[l], reduction='none')
            # defect_features = F.mse_loss(defect_bank[f'{l}'], features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_bank[f'{l}']], dim=1)
            features[l]=torch.cat([features[l], defect_bank[f'{l}']], dim=1)

        return features


