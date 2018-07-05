# -*- coding: utf-8 -*-
"""
Main file for training SQuAD reading comprehension model.
"""
import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from data_loader.SQuAD import read_dataset
from model.QANet_model import QANet
from trainer.QANet_trainer import Trainer
from util.nlp_utils import read_w2v
from util.visualize import Visualizer
from model.modules.ema import EMA


data_folder = "../../../datasets/"
parser = argparse.ArgumentParser(description='Lucy')

# dataset
parser.add_argument(
    '--train',
    default=data_folder + 'original/SQuAD/train-v1.1.json',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev',
    default=data_folder + 'original/SQuAD/dev-v1.1.json',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--train_cache',
    default=data_folder + 'processed/SQuAD/SQuAD.pkl',
    type=str, help='path of train dataset cache file')
parser.add_argument(
    '--dev_cache',
    default=data_folder + 'processed/SQuAD/SQuAD_dev.pkl',
    type=str, help='path of dev dataset cache file')
parser.add_argument(
    '--train_cache_debug',
    default=data_folder + 'processed/SQuAD/SQuAD_debug.pkl',
    type=str, help='path of train dataset cache file for debug')
parser.add_argument(
    '--dev_cache_debug',
    default=data_folder + 'processed/SQuAD/SQuAD_dev_debug.pkl',
    type=str, help='path of dev dataset cache file for debug')
parser.add_argument(
    '--validation_split',
    default=0.1, type=float,
    help='ratio of split validation data, [0.0, 1.0) (default: 0.0)')

# embedding
parser.add_argument(
    '--wemb',
    default=data_folder + 'original/Glove/glove.840B.300d.bin',
    type=str, help='path of word embedding file')
parser.add_argument(
    '--wemb_cache',
    default=data_folder + 'original/Glove/glove.840B.300d.pkl',
    type=str, help='path of word embedding cache file')
parser.add_argument(
    '--wemb_size',
    default=300, type=int,
    help='word embedding size (default: 300)')
parser.add_argument(
    '--wemb_binary',
    default=True, action='store_false',
    help='whether the word embedding file is binary')
parser.add_argument(
    '--cemb',
    default=data_folder + "original/Glove/glove.840B.300d-char.txt",
    type=str, help='path of char embedding file')
parser.add_argument(
    '--cemb_cache',
    default=data_folder + "original/Glove/glove.840B.300d-char.pkl",
    type=str, help='path of char embedding cache file')
parser.add_argument(
    '--cemb_size',
    default=300, type=int,
    help='char embedding size (default: 300)')
parser.add_argument(
    '--cemb_binary',
    default=False, action='store_true',
    help='whether the char embedding file is binary')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=16, type=int,
    help='mini-batch size (default: 16)')
parser.add_argument(
    '-e', '--epochs',
    default=32, type=int,
    help='number of total epochs (default: 32)')

# debug
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=2, type=int,
    help='only train and test a few batches when debug (devault: 2)')

# checkpoint
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--verbosity',
    default=2, type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument(
    '--save_dir',
    default='checkpoints/', type=str,
    help='directory of saved model (default: checkpoints/)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--print_freq',
    default=10, type=int,
    help='print training information frequency (default: 10 steps)')

# cuda
parser.add_argument(
    '--with_cuda',
    default=False, action='store_true',
    help='use CPU in case there\'s no GPU support')
parser.add_argument(
    '--multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')

# log & visualize
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')

# optimizer & scheduler & weight & exponential moving average
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_scheduler',
    default=True, action='store_false',
    help='whether use learning rate scheduler')
parser.add_argument(
    '--use_grad_clip',
    default=True, action='store_false',
    help='whether use gradient clip')
parser.add_argument(
    '--grad_clip',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=True, action='store_false',
    help='whether use exponential moving average')
parser.add_argument(
    '--use_early_stop',
    default=True, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=10, type=int,
    help='checkpoints for early stop')

# model
parser.add_argument(
    '--c_max_len',
    default=400, type=int,
    help='maximum context token number')
parser.add_argument(
    '--q_max_len',
    default=30, type=int,
    help='maximum question token number')
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--train_cemb',
    default=False, action='store_true',
    help='whether train char embedding or not')


def main(args):
    # show configuration
    print(args)

    # set log file
    log = sys.stdout
    if args.log_file is not None:
        log = open(args.log_file, "a")

    # set device
    if torch.cuda.is_available():
        print("device is cuda")
        print("# cuda: ", torch.cuda.device_count())
    else:
        print("device is cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load word vector
    start = datetime.now()
    specials = ["<UNK>", "<PAD>", "<SOS>", "<EOS>"]
    wv_tensor, wv_word2ix, wv_vocab, wv_dim = read_w2v(
        args.wemb,
        args.wemb_binary,
        args.wemb_size,
        specials,
        args.wemb_cache)
    cv_tensor, cv_word2ix, cv_vocab, cv_dim = read_w2v(
        args.cemb,
        args.cemb_binary,
        args.cemb_size,
        specials,
        args.cemb_cache)
    print("Time of loading word vector ", datetime.now() - start)

    # create random char tensor !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    import numpy as np
    print("before randomize: ", cv_tensor.shape)
    args.cemb_size = 64
    print("args.cemb_size now is: ", args.cemb_size)
    args.train_cemb = True
    if args.train_cemb:
        print("We will train cemb.")
    cv_tensor_rand = []
    for i in range(cv_tensor.shape[0]):
        if i == cv_word2ix["<UNK>"] or \
                i == cv_word2ix["<PAD>"] or \
                i == cv_word2ix["<SOS>"] or \
                i == cv_word2ix["<EOS>"]:
            print("specials: ", i)
            cv_tensor_rand.append([0. for _ in range(args.cemb_size)])
        else:
            cv_tensor_rand.append(
                [np.random.normal(scale=0.1) for _ in range(args.cemb_size)])
    cv_tensor = torch.FloatTensor(cv_tensor_rand)
    print("after randomize: ", cv_tensor.shape)

    # load dataset
    train_cache = args.train_cache
    dev_cache = args.dev_cache
    if args.debug:
        train_cache = args.train_cache_debug
        dev_cache = args.dev_cache_debug

    start = datetime.now()
    train_data = read_dataset(
        args.train, wv_vocab, wv_word2ix,
        cv_vocab, cv_word2ix, train_cache, args.debug)
    print("#samples in train dataset: ", train_data.__len__())
    dev_data = read_dataset(
        args.dev, wv_vocab, wv_word2ix,
        cv_vocab, cv_word2ix, dev_cache, args.debug, split="dev")
    print("#samples in dev dataset: ", dev_data.__len__())
    train_dataloader = train_data.get_dataloader(
        args.batch_size, shuffle=True, pin_memory=False)
    dev_dataloader = dev_data.get_dataloader(args.batch_size)
    print("Time of loading dataset ", datetime.now() - start)

    # construct model
    model = QANet(
        wv_tensor,
        cv_tensor,
        args.c_max_len,
        args.q_max_len,
        args.d_model,
        train_cemb=args.train_cemb,
        pad=wv_word2ix["<PAD>"])
    model.summary()
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    # set optimizer and scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    base_lr = 1.0
    optimizer = optim.Adam(
        params=parameters,
        lr=base_lr,
        betas=(args.beta1, args.beta2),
        eps=1e-7,
        weight_decay=3e-7)
    cr = args.lr / math.log2(args.lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1)
        if ee < args.lr_warm_up_num else args.lr)

    # exponential moving average
    ema = EMA(args.decay)
    if args.use_ema:
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    # set loss, metrics
    loss = F.nll_loss

    # set visdom visualizer to store training process information
    # see the training process on http://localhost:8097/
    vis = None
    if args.visualizer:
        os.system("python -m visdom.server")
        vis = Visualizer("main")

    # construct trainer
    # an identifier (prefix) for saved model
    identifier = type(model).__name__ + '_'
    trainer = Trainer(
        model, loss,
        train_data_loader=train_dataloader,
        dev_data_loader=dev_dataloader,
        train_data=args.train,
        dev_data=args.dev,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        with_cuda=args.with_cuda,
        save_dir=args.save_dir,
        verbosity=args.verbosity,
        save_freq=args.save_freq,
        print_freq=args.print_freq,
        resume=args.resume,
        identifier=identifier,
        debug=args.debug,
        debug_batchnum=args.debug_batchnum,
        lr=args.lr,
        lr_warm_up_num=args.lr_warm_up_num,
        grad_clip=args.grad_clip,
        decay=args.decay,
        visualizer=vis,
        logger=log,
        use_scheduler=args.use_scheduler,
        use_grad_clip=args.use_grad_clip,
        use_ema=args.use_ema,
        ema=ema,
        use_early_stop=args.use_early_stop,
        early_stop=args.early_stop)

    # start training!
    start = datetime.now()
    trainer.train()
    print("Time of training model ", datetime.now() - start)


if __name__ == '__main__':
    main(parser.parse_args())
