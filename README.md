# Unsupervised Deep Learning for Massive MIMO Hybrid Beamforming


In this repository you can find the simulation source code for paper title "Unsupervised Deep Learning for Massive MIMO Hybrid Beamforming". <https://ieeexplore.ieee.org/document/9439874>

## Channel model

A realistic ray-tracing channel model is considered to evaluate the proposed solution. It has been introduced by Alkhateeb, et al, in "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications". <https://arxiv.org/abs/1902.06435>


## Content

**1.DATASET.md:** all parameters related to system model such as number of users, number of antennas, etc.

**2.Codebook_ij:** designed codebook using the proposed algorithm in the paper.

**3..py files:** simulation source codes


## Dataset
**DataBase_dataSet64x8x4_130dB_0129201820.npy:** core dataset for "limited area" scenario consist of CSI, RSSI, near optimal HSHO solutions. You can find it here:
https://drive.google.com/file/d/1iXR4Zv6kBsp6NUw2bdWSGgBa6uudMKbc/view?usp=sharing

It is the core dataset with 1e4 samples. It consist of RSSI, channel, near-optimal HBF and FDP, user position. To train the DNN well enough we use 1e6 samples generted from deepMIMO channel model. The core dataset is only used for evaluate the DNN and codebook design.

## Requirements
1. torch 1.7.0
2. numpy 1.19.2

## Copyright
Feel free to use this code as a starting point for your own research project. If you do, we kindly ask that you cite the following paper: "Unsupervised Deep Learning for Massive MIMO Hybrid Beamforming". <https://ieeexplore.ieee.org/document/9439874> 

```
@ARTICLE{9439874,
  author={Hojatian, Hamed and Nadal, Jérémy and Frigon, Jean-François and Leduc-Primeau, François},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Unsupervised Deep Learning for Massive MIMO Hybrid Beamforming}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TWC.2021.3080672}}
```
Copyright (C): GNU General Public License v3.0 or later
