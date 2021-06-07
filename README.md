# **PyTorch-TransVG**

An unofficial pytorch implementation of ***"TransVG: End-to-End Visual Grounding with Transformers"***.

***paper***: https://arxiv.org/abs/2104.08541 

<img src="https://github.com/nku-shengzheliu/Pytorch-TransVG/blob/main/pipeline.PNG" width = 60% height = 60% align=center/>

Due to some implementation details, I do not guarantee that I can reproduce the performance in the paper. 

If you have any questions about the code please feel free to ask~

## Update record

* 2021.5.10
  * My model is still in training. My reproduced model performance table will be updated as soon as I finish the training.
* 2021.6.3
  * The previously trained model was very slow to converge due to the wrong setting of `image mask` in transformer encoder. I fixed this bug and re-trained now.
* 2021.6.6 Reproduced model performance:

<table>
    <tr>
        <th>Dataset</th><th>Acc@0.5</th><th>URL</th>
    </tr>
    <tr>
        <td rowspan="3">ReferItGame</td><td>val:68.07</td><td><a href="https://drive.google.com/file/d/1si1h5RPRh4WMgAvhFOtz2APKz9eJtMpY/view?usp=sharing">Google drive</a></td>
    </tr>
    <tr>
        <td>test:66.97</td><td><a href="https://pan.baidu.com/s/1QNAA6xAPlEaULrg7OqiwQQ">Baidu drive</a>[tbuq]</td>
    </tr>
</table>

## Prerequisites

Create the conda environment with the ```environment.yaml``` file:

```python
conda env create -f environment.yaml
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


Train the model using the following commands:

```
python train.py --data_root XXX --dataset {dataset_name} --gpu {gpu_id}
```

## Testing

Evaluate the model using the following commands:

```
 python train.py --data_root XXX --dataset {dataset_name} --gpu {gpu_id}  --test
```

## Acknowledgement

Thanks for the work of [DETR](https://github.com/facebookresearch/detr) and [ReSC](https://github.com/zyang-ur/ReSC). My code is based on the implementation of them.


