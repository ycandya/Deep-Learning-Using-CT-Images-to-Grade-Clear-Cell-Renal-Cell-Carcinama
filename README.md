## Deep Learning Using CT Images to Grade Clear Cell Renal Cell Carcinama: Development and Validation of a Prediction Model



This is the Pytorch implementation of Deep Learning Model  in the paper Deep Learning Using CT Images to Grade Clear Cell Renal Cell Carcinama: Development and Validation of a Prediction Model.

This implementation is based on these repositories:

- [Pytorch-classification](https://github.com/bearpaw/pytorch-classification/)

### Main Requirements

- torch == 1.0.1
- torchvision == 0.4.
- Python 3.5.7

### Training Examples

#### pre-training

- Pre-training  SeNet50/ResNet101/RegNet400/RegNet800 

  `python train.py  --selfsup 1  --model_choose 0/1/2/3`

#### developing

* Developing  SeNet50/ResNet101/RegNet400/RegNet800 for 100 epochs

  `python train.py  --selfsup 1  --epochs 100  --model_choose 0/1/2/3`

