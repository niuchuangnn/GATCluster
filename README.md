# GATCluster: Self-Supervised Gaussian-Attention Network for Image Clustering
By [Chuang Niu](https://scholar.google.com/citations?user=aoud5NgAAAAJ&hl), [Jun Zhang](https://junzhang.org/), [Ge Wang](https://www.linkedin.com/in/ge-wang-axis/), and [Jimin Liang](https://scholar.google.com/citations?user=SfkU4GEAAAAJ) 

## Introduction
This project is the Pytorch implementation of [GATCluster: Self-Supervised Gaussian-Attention Network for Image Clustering](https://arxiv.org/pdf/2002.11863.pdf).
__Accepted at ECCV 2020.__

## Installation
Assuming [Anaconda](https://www.anaconda.com/) with python 3.6, the required packages for this project can be installed as:
```shell script
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch # The latest Pytorch version (1.5.1) has been tested.
conda install -c conda-forge addict
conda install matplotlib tqdm scikit-learn requests
```
Then, clone this repo
```shell script
git clone https://github.com/niuchuangnn/GATCluster.git
cd GATCluster
```

## Run
To train the model, simply run the following commands.

For STLl10,
```shell script
python ./tools/cluster.py --config-file ./configs/stl10/gatcluster.py
```
For ImageNet10,
```shell script
python ./tools/cluster.py --config-file ./configs/imagenet10/gatcluster.py
```
The dataset will be downloaded automatically and saved to `./datasets/` when missing.
Our method is memory-efficient, a single GPU with 8G memory is enough for deep clustering.

## Visualization of attention map
Coming soon.

## Citation

```shell
@inproceedings{gatcluster2020,
  title={GATCluster: Self-Supervised Gaussian-Attention Network for Image Clustering},
  author={Niu, Chuang and Zhang, Jun and Wang, Ge and Liang, Jimin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
