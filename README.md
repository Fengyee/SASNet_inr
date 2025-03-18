# SASNet: Spatially-Adaptive Sinusoidal Neural Networks
[Haoan Feng [1]](https://fengyee.github.io), Diana Aldana [2], Tiago Novello [2], Leila De Floriani [1]
[1] University of Maryland - College Park (UMD), [2] Institute for Pure and Applied Mathematics (IMPA)

This is the official repo of "[SASNet: Spatially-Adaptive Sinusoidal Neural Networks](https://arxiv.org/abs/2503.09750)".


### Setup Environment
```bash
conda create -n sasnet python=3.9
conda activate sasnet

pip3 install torch torchvision torchaudio

# install tinycudann pytorch extension
#     prerequisit https://github.com/NVlabs/tiny-cuda-nn/tree/master?tab=readme-ov-file#requirements
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install matplotlib einops tqdm tensorboard
```


### Demo Notebook
[notebook](./demo_notebook/sasnet_tinycudann.ipynb)
