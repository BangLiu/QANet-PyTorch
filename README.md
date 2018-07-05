# QANet-pytorch

## Introduction

Re-implement [QANet](https://arxiv.org/pdf/1804.09541.pdf) with PyTorch.
Contributions are welcomed!

## Usage

Run `python QANet_main.py --batch_size 32 --with_cuda` to train model with cuda.
Run `python QANet_main.py --batch_size 32 --with_cuda --debug` to debug with small batches data.

## Structure
QANet_main.py: code for training QANet.

trainer/QANet_trainer.py: trainer.

model/QANet_model.py: defines QANet.

data_loader/SQuAD.py: SQuAD 1.1 data loader.

Other codes are utils or neural network common modules library.


## Acknowledge
1. The QANet structure implementation is mainly based on https://github.com/hengruo/QANet-pytorch.
