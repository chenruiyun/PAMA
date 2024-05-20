import cv2

from glob import glob
from einops import rearrange
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
from data import rand_perlin_2d_np
from utils import torch_seed

from typing import Union, List, Tuple
import os, sys



def mydraw(img,perlin_noise_mask,target_foreground_mask,mask,anomaly_source_img2
         ,anomaly_source_img3,anomaly_source_img4,i):
    plt.imshow(mask, cmap='gray')  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\mask\\"+str(i)+".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()  # 显示图

    plt.imshow(perlin_noise_mask, cmap='gray')  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\perlin_noise_mask\\" + str(i) + ".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()  # 显示图

    plt.imshow(target_foreground_mask,cmap='gray')  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\target_foreground_mask\\" + str(i) + ".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()  # 显示图

    plt.imshow(anomaly_source_img2/255.0, cmap='gray')  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\texture\\" + str(i) + ".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()  #

    plt.imshow(anomaly_source_img3/255.0)  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\perlin_text\\" + str(i) + ".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()  #

    plt.imshow(img / 255.0)  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\source_img\\" + str(i) + ".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()

    plt.imshow(anomaly_source_img4/255.0)  # cmap参数可以设置颜色映射（colormap），这里使用'viridis'作为例子
    plt.axis('off')
    save_path = r"C:\Users\think\Desktop\ppt\capsule\final\\" + str(i) + ".png"  # 更改为您的路径和文件名
    plt.savefig(save_path)
    plt.show()  #

    i=i+1

    return i
mvtec_path=r'E:\memseg2\pama_cry\datasets\MVTec'
generated_data_path=r'E:\generatedata\generated_dataset'


class MVTec_Anomaly_Detection(Dataset):
    def __init__(self,sample_name,length=5000,anomaly_id=None,recon=False):
        self.recon=recon
        self.good_path='%s/%s/train/good'%(mvtec_path,sample_name)
        self.good_files=[os.path.join(self.good_path,i) for i in os.listdir(self.good_path)]
        self.root_dir = '%s/%s'%(generated_data_path,sample_name)
        self.anomaly_names=os.listdir(self.root_dir)
        if anomaly_id!=None:
            self.anomaly_names=self.anomaly_names[anomaly_id:anomaly_id+1]
            print('training subsets',self.anomaly_names)
        l=len(self.anomaly_names)
        self.anomaly_num = l
        self.img_paths=[]
        self.mask_paths=[]
        for idx,anomaly in enumerate(self.anomaly_names):
            img_path=[]
            mask_path=[]
            for i in range(min(len(os.listdir(os.path.join(self.root_dir,anomaly,'mask'))),500)):
                img_path.append(os.path.join(self.root_dir,anomaly,'image','%d.jpg'%i))
                mask_path.append(os.path.join(self.root_dir,anomaly,'mask','%d.jpg'%i))
            self.img_paths.append(img_path.copy())
            self.mask_paths.append(mask_path.copy())
        for i in range(l):
            print(len(self.img_paths[i]),len(self.mask_paths[i]))
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])
        self.length=length
        if self.length is None:
            self.length=len(self.good_files)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random()>0.5:
            image=self.loader(Image.open(self.good_files[idx%len(self.good_files)]).convert('RGB'))
            mask=torch.zeros((1,image.size(-2),image.size(-1)))
            has_anomaly = np.array([0], dtype=np.float32)
            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': -1}
            if self.recon:
                sample['source']=image
        else:
            anomaly_id=random.randint(0,self.anomaly_num-1)
            img_path=self.img_paths[anomaly_id][idx% len(self.mask_paths[anomaly_id])]
            image = self.loader(Image.open(img_path).convert('RGB'))
            mask_path = self.mask_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
            mask = self.loader(Image.open(mask_path).convert('L'))
            mask=(mask>0.5).float()
            if mask.sum()==0:
                has_anomaly = np.array([0], dtype=np.float32)
                anomaly_id=-1
            else:
                has_anomaly = np.array([1], dtype=np.float32)
            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': anomaly_id}
            if self.recon:
                img_path = self.img_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
                img_path=img_path.replace('image','recon')
                ori_image = self.loader(Image.open(img_path).convert('RGB'))
                sample['source']=ori_image
        return sample
class pamaDataset(Dataset):

    def __init__(
        self, datadir: str, target: str, train: bool, to_memory: bool = False,
        resize: Tuple[int, int] = (224,224),
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
    ):
        # mode
        self.train = train 
        self.to_memory = to_memory
        self.global_i=0
        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*/*' if train else 'test/*/*'))#现在只有正常的


        #load yichang image
        self.n_anomaly=10
        self.ramdn_seed=42

        #加异常的进去
        if train:
            self.outlier_data = self.split_outlier()
            # self.outlier_data .sort()
            # outlier_data=os.path.join(self.datadir, self.target,outlier_data)
            self.file_list=self.file_list+self.outlier_data
        
        # load texture image file list
        if texture_source_dir:
            self.texture_source_file_list = glob(os.path.join(texture_source_dir,'*/*'))
            
        # synthetic anomaly
        if train:
            self.transparency_range = transparency_range
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            self.structure_grid_size = structure_grid_size
        
        # transform ndarray into tensor
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean = (0.485, 0.456, 0.406),
            #     std  = (0.229, 0.224, 0.225)
            # )
        ])
        self.epoch = False
        # sythetic anomaly switch
        self.anomaly_switch = False
        
    def __getitem__(self, idx):
        
        file_path = self.file_list[idx]
        
        # image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
        
        # target
        #'ok' in self.file_list[idx] or
        if  'good' in self.file_list[idx]:
        # if 'ok' in self.file_list[idx]:
            # print(self.file_list[idx] )
            target = 0
        else:
            target = 1
        
        # mask 'ok' in file_path
        if 'good' in file_path:
        # if 'ok' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = cv2.imread(
                file_path.replace('test','ground_truth').replace('.png', '_mask.png').replace('.bmp','.bmp').replace('image','mask'),
                cv2.IMREAD_GRAYSCALE
            )
            mask = cv2.resize(mask, dsize=(self.resize[1], self.resize[0])).astype(np.bool).astype(np.int)

        # anomaly source
        if self.epoch:
            if not self.to_memory and self.train:
                 if self.anomaly_switch:
                     img, mask = self.generate_anomaly(img=img)
                     target = 1
                     self.anomaly_switch = False
                 else:
                   self.anomaly_switch = True

        # convert ndarray into tensor
        img = self.transform(img)
        mask = torch.Tensor(mask).to(torch.int64)
        
        return img, mask, target
        
        
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
        
    def generate_anomaly(self, img: np.ndarray) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask
            
        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation, 
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid 
            and randomly arranged to obtain the disordered image  𝐼 
            
        step 3. blending image and anomaly source
        '''
        
        # step 1. generate mask
        
        ## target foreground mask
        h, w, c = img.shape
        if self.target=='02' or 'sdd' in self.target:
            target_foreground_mask= np.ones((h,w))
        else:
          target_foreground_mask = self.generate_target_foreground_mask(img=img)
        if self.target=='pill' or self.target=='01' or self.target=='now':
            target_foreground_mask=1-target_foreground_mask

        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask()
        
        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)
        
        # step 2. generate texture or structure anomaly
        
        ## anomaly source
        anomaly_source_img = self.anomaly_source(img=img)
        anomaly_source_img2=anomaly_source_img.copy()
        ## mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
        anomaly_source_img3 = anomaly_source_img.copy()
        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
        anomaly_source_img4 = anomaly_source_img.copy()
        # self.global_i=mydraw(img,perlin_noise_mask, target_foreground_mask, mask, anomaly_source_img2
        #        ,anomaly_source_img3, anomaly_source_img4, self.global_i)

        return (anomaly_source_img.astype(np.uint8), mask)

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)

        return target_foreground_mask
    
    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray) -> np.ndarray:
        p = np.random.uniform()
        if p < 0.5:
        #     # TODO: None texture_source_file_list
            anomaly_source_img = self._texture_source()
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self) -> np.ndarray:
        idx = np.random.choice(len(self.texture_source_file_list))
        # print(len(self.texture_source_file_list))
        # print(idx)
        texture_source_img = cv2.imread(self.texture_source_file_list[idx])
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=(self.resize[1], self.resize[0])).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        assert self.resize[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = self.resize[1] // self.structure_grid_size
        grid_h = self.resize[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)

    def split_outlier(self):
        generated_root=r'E:\generatedata\generated_dataset'
        outlier_data_dir=os.path.join(generated_root,self.target)
        tmp=outlier_data_dir
        # outlier_data_dir = os.path.join(self.datadir,self.target, 'test')
        # tmp = os.path.join(self.datadir, self.target)
        outlier_classes = os.listdir(outlier_data_dir)
        outlier_data = list()

        for cl in outlier_classes:
            # if cl == 'good':
            if cl =='ok'or cl=='good':
                continue
            generated_data_dir=os.path.join(outlier_data_dir, cl,'image')
            outlier_file=os.listdir(generated_data_dir)
            # outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))

            for file in outlier_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
                    # outlier_data.append(tmp+'/test/' + cl + '/' + file)
                    outlier_data.append(generated_data_dir+'/'+file)

        if self.n_anomaly > len(outlier_data)/2:
            print(len(outlier_data))
            print("Number of outlier data in training set should less than half of outlier datas!")
            sys.exit()
        np.random.RandomState(self.ramdn_seed).shuffle(outlier_data)
        # return outlier_data
        if self.train:
            return outlier_data[0:self.n_anomaly]
        else:
            return outlier_data[self.n_anomaly:]

        return outlier_data
# for i in range(16):
#     plt.imshow(inputs[i].permute(1,2,0).detach().cpu())
#     plt.show()
#     plt.imshow(masks[i].unsqueeze(0).permute(1, 2, 0).detach().cpu())
#     plt.show()
#     plt.imshow(outputs[i][1].unsqueeze(0).permute(1, 2, 0).detach().cpu())
#     plt.show()
