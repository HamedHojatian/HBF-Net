## Unsupervised Deep Learning for Massive MIMO Hybrid Beamforming



In this repository you can find the simulation source code for paper title "Unsupervised Deep Learning for Massive MIMO Hybrid Beamforming". <https://arxiv.org/abs/2007.00038>

A realistic ray-tracing channel model is considered to evaluate the proposed solution. It has been introduced by Alkhateeb, et al, in "DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications". <https://arxiv.org/abs/1902.06435>


The repository consist of:

**1.DATASET.md:** all parameters related to system model such as number of users, number of antennas, etc.

**2.Codebook_ij:** design codebook using algorithm proposed in the paper.

**3..py files:** simulation source codes

**DataBase_dataSet64x8x4_130dB_0129201820.npy:** core dataset for "limited area" scenario consist of CSI, RSSI, near optimal HSHO solutions. You can find it here:
https://drive.google.com/file/d/1iXR4Zv6kBsp6NUw2bdWSGgBa6uudMKbc/view?usp=sharing

Requirements:
1. torch 1.6.0
2. numpy 1.19.2

