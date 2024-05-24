# PAMA
Implementation for PAMA: Patch Matching for Few-shot Industrial Defect Detection

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
--yaml_config
./configs/grid.yaml

# Note
Both the memory bank and the network weights are objects that need to be saved

