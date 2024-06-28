# PAMA
Implementation for PAMA: \
"Patch Matching for Few-shot Industrial Defect Detection"  \
please refer to: \
Chen, Ruiyun, Guitao Yu, Zhen Qin, Kangkang Song, Jianfei Tu, Xianliang Jiang, Dan Liang, and Chengbin Peng. “Patch Matching for Few-shot Industrial Defect Detection.” IEEE Transactions on Instrumentation and Measurement (2024).

```
@ARTICLE{10555421,
  author={Chen, Ruiyun and Yu, Guitao and Qin, Zhen and Song, Kangkang and Tu, Jianfei and Jiang, Xianliang and Liang, Dan and Peng, Chengbin},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Patch Matching for Few-shot Industrial Defect Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Image reconstruction;Defect detection;Feature extraction;Image segmentation;Anomaly detection;Memory management;Training;Defect detection;Matching;Space efficient memory bank;Multi-view attention},
  doi={10.1109/TIM.2024.3413170}}

```
# Environments
anomalib==0.3.7 \
opencv-python==4.6.0\
einops==0.5.0\
timm==0.5.4

# Datasets
MvtecAD ([MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)) \
BTAD ([VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization](http://suo.nz/2JEGEi))\
SDD ([Segmentation-Based Deep-Learning Approach for Surface-Defect Detection](https://opendatalab.com/OpenDataLab/KolektorSDD))\
AnomalyDiffusion([AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model](https://github.com/sjtuplayer/anomalydiffusion.
))      [Generated data](https://drive.google.com/file/d/1GaA3oGnYYNK62FagQubQKS5YcgmCG8PT)

# Run
## Example 
python main.py --yaml_config ./configs/grid.yaml

# Note
During the inference process, the grid weight file located at `./saved_model/grid/best_model.pt` and the grid memory bank weight file located at `./saved_model/grid/memory_bank.pt` are required.


