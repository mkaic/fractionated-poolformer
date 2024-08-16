
# The idea
PoolFormer, but different channels have different pooling kernel sizes, and there's no downsampling.

# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d --gpus all -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```

# Repo structure
Implementations are in `src`, training script is in `scripts` along with a few sanity-checks. The training script expects CIFAR-10/100 to be in a folder called `data`.