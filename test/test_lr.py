# -*- coding: utf-8 -*-
"""
This is used to check how the learning rate changes during training.
"""
import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# data settings
num_train_samples = 86643
batch_size = 32
epoch = 30
parameters = {torch.zeros(3, 3, requires_grad=True)}

# optimizer
lr = 0.001
lr_warm_up_num = 1000
decay = 0.9999
base_lr = 1.0
optimizer = optim.Adam(
    params=parameters,
    lr=base_lr,
    betas=(0.8, 0.999),
    eps=1e-7,
    weight_decay=3e-7)

# scheduler
cr = lr / math.log2(lr_warm_up_num)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda ee: cr * math.log2(ee + 1)
    if ee < lr_warm_up_num else lr)

# update lr during training
unused = True
global_step = 0
lrs_warm_up = []
lrs = []
lrs_epoch = []
batch_step = math.ceil(num_train_samples / batch_size)
for e in range(0, epoch):
    for step in range(0, batch_step):
        if global_step < lr_warm_up_num - 1:
            scheduler.step()
            lrs_warm_up.append(optimizer.param_groups[0]['lr'])
        # after warm up stage, fix scheduler
        if global_step >= lr_warm_up_num - 1 and unused:
            optimizer.param_groups[0]['initial_lr'] = lr
            for g in optimizer.param_groups:
                g['lr'] = lr
                print("glr is: ", lr)
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, decay)
            unused = False
        # print("Learning rate: {}".format(scheduler.get_lr()))
        # print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))
        # print(len(optimizer.param_groups))
        lrs.append(optimizer.param_groups[0]['lr'])
        global_step += 1
    lrs_epoch.append(optimizer.param_groups[0]['lr'])

# show how lr is changing during warm up
x = range(0, lr_warm_up_num - 1)
plt.plot(x, lrs_warm_up)
plt.show()
plt.close()

# show how lr is changing with each step
x = range(0, epoch * batch_step)
plt.plot(x, lrs)
plt.show()
plt.close()

# show how lr is changing with each epoch
x = range(0, epoch)
plt.plot(x, lrs_epoch)
plt.show()
plt.close()
