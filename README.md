# iControl3D: An Interactive System for Controllable 3D Scene Generation (ACM MM 2024)

[Xingyi Li](https://xingyi-li.github.io/)<sup>1,2</sup>,
[Yizheng Wu](https://scholar.google.com/citations?user=0_iF4jMAAAAJ&hl=en)<sup>1,2</sup>,
[Jun Cen](https://cen-jun.com/)<sup>2</sup>,
[Juewen Peng](https://juewenpeng.github.io/)<sup>2</sup>,
[Kewei Wang](https://scholar.google.com/citations?user=fW7pUGMAAAAJ&hl=en)<sup>1,2</sup>,
[Ke Xian](https://kexianhust.github.io/)<sup>1</sup>,
[Zhe Wang](https://wang-zhe.me/)<sup>3</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1\*</sup>,
[Guosheng Lin](https://guosheng.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Nanyang Technological University, <sup>3</sup>SenseTime Research

[Paper](https://github.com/xingyi-li/iControl3D/) | [arXiv](https://github.com/xingyi-li/iControl3D/) | [Video](https://github.com/xingyi-li/iControl3D/) | [Supp](https://github.com/xingyi-li/iControl3D/) | [Poster](https://github.com/xingyi-li/iControl3D/)

This repository contains the official PyTorch implementation of our ACM MM 2024 paper "iControl3D: An Interactive System for Controllable 3D Scene Generation".

## Environment Setup

First install dependencies: 
```shell
conda create -n icontrol3d python=3.10
conda activate icontrol3d
conda install pytorch=1.13.0 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install scipy scikit-image
conda install -c conda-forge diffusers transformers ftfy accelerate
pip install opencv-python
pip install -U gradio
pip install pytorch-lightning==1.7.7 einops==0.4.1 omegaconf==2.2.3
pip install timm

# Install diffusers
git clone https://github.com/takuma104/diffusers.git
cd diffusers
git checkout 9a37409663a53f775fa380db332d37d7ea75c915
pip install .

# Update transformers and huggingface_hub
pip install git+https://github.com/huggingface/transformers
pip install -U huggingface_hub

# Pytorch3D
conda install -c iopath iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

# skylibs
pip install --upgrade skylibs
conda install -c conda-forge openexr-python openexr
conda install -c conda-forge pyshtools

# Grounded-Segment-Anything
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

Follow [https://github.com/haofanwang/ControlNet-for-Diffusers](https://github.com/haofanwang/ControlNet-for-Diffusers) and download pipeline_stable_diffusion_controlnet_inpaint.py to enable ControlNet for `diffusers`:
```
# assume you already know the absolute path of installed diffusers
cp pipeline_stable_diffusion_controlnet_inpaint.py PATH/pipelines/stable_diffusion
```
Then, you need to import this new added pipeline in corresponding files
```
PATH/pipelines/stable_diffusion/__init__.py
PATH/pipelines/__init__.py
PATH/__init__.py
```

Last but not least, as per ([https://github.com/haofanwang/ControlNet-for-Diffusers/issues/6](https://github.com/haofanwang/ControlNet-for-Diffusers/issues/6)), to use any control model already present in ControlNet models, the way to do it is:

Download the models and annotators from the controlnet huggingface repo (https://huggingface.co/lllyasviel/ControlNet)[https://huggingface.co/lllyasviel/ControlNet] and place it under models folder. Then convert the models which can be used with the pipeline:

```shell
cd diffusers
python ./scripts/convert_controlnet_to_diffusers.py --checkpoint_path ./models/control_sd15_***.pth --dump_path ../controlnet_models/control_sd15_*** --device cpu
```

For Grounded-SAM: 
```shell
cd lib/grounded_sam

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Usage
```
conda activate icontrol3d
# scribble
python app_controlnet_inpaint.py
# depth
# python app_controlnet_inpaint_depth.py
# hed
# python app_controlnet_inpaint_hed.py
# seg
# python app_controlnet_inpaint_seg.py
# canny
# python app_controlnet_inpaint_canny.py
# mlsd
# python app_controlnet_inpaint_mlsd.py
```
You can add `--outdoor` and adjust parameters like `--box_threshold` to enable the ability to handle outdoor scenes. Please refer to `lib/utils/opt.py` for more information.

After this, you can use `nerfstudio` to train a NeRF and render videos.


## Acknowledgement
This code is built on [stablediffusion-infinity](https://github.com/lkwq007/stablediffusion-infinity), [Text2Room](https://github.com/lukasHoel/text2room) and many other projects. We would like to acknowledge them for making great code openly available for us to use.