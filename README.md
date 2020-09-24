# GATCluster: Self-Supervised Gaussian-Attention Network for Image Clustering
By [Chuang Niu](https://scholar.google.com/citations?user=aoud5NgAAAAJ&hl), [Jun Zhang](https://junzhang.org/), [Ge Wang](https://www.linkedin.com/in/ge-wang-axis/), and [Jimin Liang](https://scholar.google.com/citations?user=SfkU4GEAAAAJ) 

## Introduction
This project is the Pytorch implementation of the [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700732.pdf)
__at ECCV 2020.__

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
Before visualizing the attention maps, 
a model should be trained and set the corresponding path in `./tools/visualize_att_map.py`.
You can also use our trained model on STL10 at [here](https://drive.google.com/uc?export=download&id=1LXfoWhLpM7yiVJy_POkdOImHfkDjz1xI),
and place it in `./results/stl10/gatcluster/`. This model is reproduced and its accuracy is slightly better than the best result reported in our paper.

Then, run
```shell script
python ./tools/visualize_att_map.py
```
The results will be saved in `./results/stl10/att_maps/` as:

<table align='c'>

<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/0.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/1.png"></td>
</tr>

<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/2.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/3.png"></td>
</tr>
<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/4.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/5.png"></td>
</tr>
<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/6.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/7.png"></td>
</tr>
<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/8.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/stl10/att_maps/9.png"></td>
</tr>

</table>

For ImageNet10, our trained model can be downloaded at [here](https://drive.google.com/uc?export=download&id=1F-_YbuszPSAO_eWCzeJeCTOfgb0j5Vvl),
and placed into `./results/imagenet10/gatcluster/` for visualization.

Then, run
```shell script
python ./tools/visualize_att_map_imagenet10.py
```

The results will be saved in `./results/imagenet10/att_maps/` as:

<table align='c'>

<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/0.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/1.png"></td>
</tr>

<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/2.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/3.png"></td>
</tr>
<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/4.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/5.png"></td>
</tr>
<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/6.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/7.png"></td>
</tr>
<tr>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/8.png"></td>
<td><img  height="120" src="https://github.com/niuchuangnn/GATCluster/blob/master/results/imagenet10/att_maps/9.png"></td>
</tr>

</table>

## Citation

```shell
@inproceedings{gatcluster2020,
  title={GATCluster: Self-Supervised Gaussian-Attention Network for Image Clustering},
  author={Niu, Chuang and Zhang, Jun and Wang, Ge and Liang, Jimin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
