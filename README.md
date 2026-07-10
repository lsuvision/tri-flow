# Tri-Flow

PyTorch implementation of triangular consistency described in:

**Triangular Consistency as A Universal Constraint For Learning Optical Flow**
[[`arXiv`](https://arxiv.org/abs/2606.19938)]

<div align="center">
  <img width="100%" alt="Triangular Consistency" src=".github/triangular.png">
</div>


## Usage

## Self-Supervised Optical Flow

## Unsupervised Optical Flow

## Supervised Optical Flow

We applied the tri-flow method to the [RAFT](https://github.com/princeton-vl/raft) using a pre-trained model on the FlyingChairs and FlyingThings3D datasets and applying the augmentation method described in the paper for the rest of the training pipeline. The code implementation is present in the "Supervised" folder.

## License

This repository is released under the [MIT LICENSE](LICENSE)

## Citation

If this work is useful in your research, please consider citing it and giving the repository a star :star:

```
@inproceedings{Xiao2026Tri-Flow,
  title={Triangular Consistency as A Universal Constraint For Learning Optical Flow},
  author={Xiao, Yi, and Rodriguez Coronel, Carlos and Zhan, Jing and Oskouie, Haniyeh Ehsani and Wong, Alex and Lao, Dong},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
