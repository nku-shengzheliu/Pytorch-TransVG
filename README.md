# **PyTorch-TransVG**

An unofficial pytorch implementation of ***"TransVG: End-to-End Visual Grounding with Transformers"***.

***paper***: https://arxiv.org/abs/2104.08541 

<img src="https://github.com/nku-shengzheliu/Pytorch-TransVG/blob/main/pipeline.PNG" width = 60% height = 60% align=center/>

My model is still in training. Due to some implementation details, I do not guarantee that I can reproduce the performance in the paper. My own reproduced model performance table will be updated as soon as I finish the training.

Also, if you have any questions about the code please feel free to ask.



## Prerequisites

Create the conda environment with the ```environment.yml``` file:

```python
conda env create -f environment.yml
```

Activate the environment with:

```python
conda activate transvg
```

## Installation

1. Please refer to [ReSC](https://github.com/zyang-ur/ReSC), and follow the steps to **Prepare the submodules and associated data**:

* RefCOCO, RefCOCO+, RefCOCOg, ReferItGame Dataset.
* Dataset annotations, which stored in `./data`

2. Please refer to [DETR](https://github.com/facebookresearch/detr) and download model weights, I used the DTER model with ResNet50, which reached an AP of 42.0 at COCO2017. Please store it in `./saved_models/detr-r50-e632da11.pth`

## Training


Train our model using the following commands:

```
python train.py --data_root XXX --dataset {dataset_name} --gpu {gpu_id}
```

## Testing

Evaluate our model using the following commands:

```
 python train.py --data_root XXX --dataset {dataset_name} --gpu {gpu_id}  --test
```



