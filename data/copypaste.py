import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import logging
#from skimage.filters import gaussian
from skimage.measure import label, regionprops
import torch
import matplotlib.pyplot as plt
import cv2
def mydraw(img,perlin_noise_mask,target_foreground_mask,mask,anomaly_source_img2
         ,anomaly_source_img3,anomaly_source_img4,i):
    plt.imshow(mask, cmap='gray')  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\seen\\"+str(i)+".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()  # 显示图
    i=i+1

    return i

# def dynamic_copy_paste(images_sup, labels_sup,targets,outlier_data_image,outlier_data_mask,outlier_data):
#     # images_sup, paste_imgs = torch.chunk(images_sup,2,dim=0)
#     # labels_sup, paste_labels = torch.chunk(labels_sup,2,dim=0)
#     # labels_sup, paste_labels = labels_sup.squeeze(1), paste_labels.squeeze(1)
# #iamges分为两半  标签也分为两半
#     compose_imgs = []
#     compose_labels = []
#     compose_targets=[]
#     anomaly_switch = False
#     for idx in range(images_sup.shape[0]):
#         index = np.random.randint(0, outlier_data_image.shape[0])
#         paste_label = outlier_data_mask[index]
#         image_sup = images_sup[idx]
#         label_sup = labels_sup[idx]
#         target=targets[idx]
#         if torch.sum(label_sup) > 1: #伪异常图
#             compose_imgs.append(image_sup.unsqueeze(0))
#             compose_labels.append(label_sup.unsqueeze(0))
#             compose_targets.append(target)
#         elif torch.sum(label_sup) == 0 and anomaly_switch:
#             paste_img = outlier_data_image[index]
#             # alpha = torch.zeros_like(paste_label).int()
#             alpha=paste_label
#             # alpha = (alpha > 0).int()
#             angle = np.random.randint(0, 360)
#             tx = np.random.randint(-10, 10)
#             ty = np.random.randint(-10, 10)
#             height, width = paste_img.shape[1:3]
#             center = (width // 2, height // 2)
#             rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#             paste_img=paste_img.detach().cpu().numpy()
#
#             paste_label=paste_label.detach().cpu().numpy()
#             rotated_image = cv2.warpAffine(paste_img.transpose(1,2,0), rotation_matrix, (width, height))
#             rotated_mask = cv2.warpAffine(paste_label, rotation_matrix, (width, height))
#             translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
#             translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
#             translated_mask = cv2.warpAffine(rotated_mask, translation_matrix, (width, height))
#             paste_img1=torch.from_numpy(translated_image).cuda().permute(2,0,1)
#             alpha=torch.from_numpy(translated_mask).cuda()
#             paste_label1=torch.from_numpy(translated_mask).cuda()
#             gray_tensor = image_sup.mean(dim=0, keepdim=True)
#             threshold = 0.5
#             binary_tensor = torch.where(gray_tensor > threshold, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
#             if 'pill' in outlier_data[0]:
#                 binary_tensor=1-binary_tensor
#             alpha = torch.logical_and((1 - binary_tensor), alpha).long()
#
#
#             compose_img = alpha * paste_img1 + (1-alpha) * image_sup
#             compose_label = alpha*paste_label1 + (1-alpha) * label_sup
#             compose_imgs.append(compose_img.unsqueeze(0))
#             compose_labels.append(compose_label)
#             if torch.sum(compose_label) ==0:
#                 compose_targets.append(target)
#             else:
#                 compose_targets.append(1 - target)
#                 anomaly_switch=False
#         else:
#             anomaly_switch=True
#             compose_imgs.append(image_sup.unsqueeze(0))
#             compose_labels.append(label_sup.unsqueeze(0))
#             compose_targets.append(target)
#     compose_imgs = torch.cat(compose_imgs,dim=0)
#     compose_labels = torch.cat(compose_labels,dim=0)
#     compose_targets=torch.tensor(compose_targets)
#     return compose_imgs, compose_labels.long(),compose_targets

import numpy as np
import torch
import cv2


def dynamic_copy_paste(images_sup, labels_sup, targets, outlier_data_image, outlier_data_mask, outlier_data):
    compose_imgs = []
    compose_labels = []
    compose_targets = []
    anomaly_switch = False

    for idx in range(images_sup.shape[0]):
        index = np.random.randint(0, outlier_data_image.shape[0])
        paste_label = outlier_data_mask[index]
        image_sup = images_sup[idx]
        label_sup = labels_sup[idx]
        target = targets[idx]

        if torch.sum(label_sup) > 1:  # 伪异常图
            compose_imgs.append(image_sup.unsqueeze(0))
            compose_labels.append(label_sup.unsqueeze(0))
            compose_targets.append(target)

        elif torch.sum(label_sup) == 0 and anomaly_switch:
            paste_img = outlier_data_image[index]
            alpha = paste_label
            angle = np.random.randint(0, 360)
            scale = np.random.uniform(0.8, 1.2)  # 随机缩放
            tx = np.random.randint(-image_sup.shape[2] // 4, image_sup.shape[2] // 4)
            ty = np.random.randint(-image_sup.shape[1] // 4, image_sup.shape[1] // 4)
            height, width = paste_img.shape[1:3]
            center = (width // 2, height // 2)

            # 生成旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            paste_img = paste_img.detach().cpu().numpy()
            paste_label = paste_label.detach().cpu().numpy()

            rotated_image = cv2.warpAffine(paste_img.transpose(1, 2, 0), rotation_matrix, (width, height))
            rotated_mask = cv2.warpAffine(paste_label, rotation_matrix, (width, height))

            # 生成平移矩阵
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
            translated_mask = cv2.warpAffine(rotated_mask, translation_matrix, (width, height))

            paste_img1 = torch.from_numpy(translated_image).cuda().permute(2, 0, 1)
            alpha = torch.from_numpy(translated_mask).cuda()
            paste_label1 = torch.from_numpy(translated_mask).cuda()

            gray_tensor = image_sup.mean(dim=0, keepdim=True)
            threshold = 0.5
            binary_tensor = torch.where(gray_tensor > threshold, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
            if 'pill' in outlier_data[0]:
                binary_tensor = 1 - binary_tensor
            alpha = torch.logical_and((1 - binary_tensor), alpha).long()

            compose_img = alpha * paste_img1 + (1 - alpha) * image_sup
            compose_label = alpha * paste_label1 + (1 - alpha) * label_sup
            compose_imgs.append(compose_img.unsqueeze(0))
            compose_labels.append(compose_label)
            if torch.sum(compose_label) == 0:
                compose_targets.append(target)
            else:
                compose_targets.append(1 - target)
                anomaly_switch = False
        else:
            anomaly_switch = True
            compose_imgs.append(image_sup.unsqueeze(0))
            compose_labels.append(label_sup.unsqueeze(0))
            compose_targets.append(target)

    compose_imgs = torch.cat(compose_imgs, dim=0)
    compose_labels = torch.cat(compose_labels, dim=0)
    compose_targets = torch.tensor(compose_targets)

    return compose_imgs, compose_labels.long(), compose_targets

# plt.imshow(compose_label.unsqueeze(0).permute(1,2,0).detach().cpu())
# plt.show()