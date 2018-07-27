# QANet-PyTorch

## Introduction

Re-implement [QANet](https://arxiv.org/pdf/1804.09541.pdf) with PyTorch.
Contributions are welcomed!

## Usage

Run `python3 QANet_main.py --batch_size 32 --epochs 20 --with_cuda --use_ema ` to train model with cuda.
Run `python3 QANet_main.py --batch_size 32 --epochs 20 --with_cuda --use_ema --debug` to debug with small batches data.

## **Performance**

Without ema:
EM: 74
F1: 63

With ema:
Still testing

## Structure
QANet_main.py: code for training QANet.

trainer/QANet_trainer.py: trainer.

model/QANet_model.py: defines QANet.

data_loader/SQuAD.py: SQuAD 1.1 and 2.0 data loader.

Other codes are utils or neural network common modules library.


## Acknowledge
1. The QANet structure implementation is mainly based on https://github.com/hengruo/QANet-pytorch and https://github.com/andy840314/QANet-pytorch-.
2. For a TensorFlow implementation, please refer to https://github.com/NLPLearn/QANet.
