# -*- coding: utf-8 -*-
"""
Trainer file for SQuAD dataset.
"""
import os
import shutil
import time
import torch
import json
from datetime import datetime
import torch.optim as optim
from .metric import em_by_begin_end_index, f1_by_begin_end_index
from .config import *
from model.modules.predicting import boundary2idx


class Trainer(object):

    def __init__(self, model, loss,
                 train_data_loader, dev_data_loader,
                 train_data, dev_data,
                 optimizer, scheduler, epochs, with_cuda,
                 save_dir, verbosity=2, save_freq=1, print_freq=10,
                 resume=False, identifier='',
                 debug=False, debug_batchnum=2,
                 visualizer=None, logger=None,
                 grad_clip=5.0, decay=0.9999,
                 lr=0.001, lr_warm_up_num=1000,
                 use_scheduler=False, use_grad_clip=False,
                 use_ema=False, ema=None,
                 use_early_stop=False, early_stop=10):
        # for evaluate
        with open(dev_data) as dataset_file:
            dataset_json = json.load(dataset_file)
            self.dev_dataset = dataset_json['data']
        with open(train_data) as dataset_file:
            dataset_json = json.load(dataset_file)
            self.train_dataset = dataset_json['data']

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.verbosity = verbosity
        self.identifier = identifier
        self.visualizer = visualizer
        self.with_cuda = with_cuda
        self.device = torch.device("cuda" if with_cuda else "cpu")

        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.is_debug = debug
        self.debug_batchnum = debug_batchnum
        self.logger = logger
        self.unused = True  # whether scheduler has been updated

        self.lr = lr
        self.lr_warm_up_num = lr_warm_up_num
        self.decay = decay
        self.use_scheduler = use_scheduler
        self.scheduler = scheduler
        self.use_grad_clip = use_grad_clip
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema = ema
        self.use_early_stop = use_early_stop
        self.early_stop = early_stop

        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        self.best_em = 0
        self.best_f1 = 0
        if resume:
            self._resume_checkpoint(resume)
            self.model = self.model.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

    def train(self):
        patience = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if self.use_early_stop:
                if result["f1"] < self.best_f1 and result["em"] < self.best_em:
                    patience += 1
                    if patience > self.early_stop:
                        print("Perform early stop!")
                        break
                else:
                    patience = 0

            is_best = False
            if result["f1"] > self.best_f1:
                is_best = True
            if result["f1"] == self.best_f1 and result["em"] > self.best_em:
                is_best = True
            self.best_f1 = max(self.best_f1, result["f1"])
            self.best_em = max(self.best_em, result["em"])

            if epoch % self.save_freq == 0:
                self._save_checkpoint(
                    epoch, result["f1"], result["em"], is_best)

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)

        # initialize
        global_loss = 0.0
        global_em = 0.0
        global_f1 = 0.0
        last_step = self.step - 1
        last_time = time.time()

        # train over batches
        for batch_idx, batch in enumerate(self.train_data_loader):
            # get batch
            (question_lengths,
             question_wids,
             question_cids,
             context_lengths,
             context_wids,
             context_cids,
             context_sent_lengths,
             context_sent_wids,
             context_sent_cids,
             answer_spos,
             answer_tpos,
             answer_tpos_in_sent) = batch
            batch_num, question_len = question_wids.size()
            print("#samples: ", batch_idx * batch_num,
                  file=self.logger, flush=True)
            _, context_len = context_wids.size()
            _, sent_num, sent_len = context_sent_wids.size()
            targets = torch.rand(batch_num).long()
            sent_boundary_num = int((sent_len + 1) * sent_len / 2)
            for i in range(batch_num):
                targets[i] = boundary2idx(
                    answer_tpos_in_sent[i, 0].item(),
                    answer_tpos_in_sent[i, 1].item(),
                    sent_len).tolist() + sent_boundary_num * answer_spos[i]
            targets = targets.to(self.device)
            question_lengths = question_lengths.to(self.device)
            question_wids = question_wids.to(self.device)
            question_cids = question_cids.to(self.device)
            context_lengths = context_lengths.to(self.device)
            context_wids = context_wids.to(self.device)
            context_cids = context_cids.to(self.device)
            context_sent_lengths = context_sent_lengths.to(self.device)
            context_sent_wids = context_sent_wids.to(self.device)
            context_sent_cids = context_sent_cids.to(self.device)
            answer_spos = answer_spos.to(self.device)
            answer_tpos = answer_tpos.to(self.device)
            answer_tpos_in_sent = answer_tpos_in_sent.to(self.device)

            # calculate loss and update model
            self.model.zero_grad()
            p1, p2 = self.model(
                context_wids,
                context_cids,
                question_wids,
                question_cids)
            y1, y2 = answer_tpos[:, 0], answer_tpos[:, 1]
            loss1 = self.loss(p1, y1, size_average=True)
            loss2 = self.loss(p2, y2, size_average=True)
            loss = (loss1 + loss2) / 2
            loss.backward()
            self.optimizer.step()

            # update learning rate
            if self.use_scheduler:
                self.scheduler.step()
                if self.step >= self.lr_warm_up_num - 1 and self.unused:
                    self.optimizer.param_groups[0]['initial_lr'] = self.lr
                    self.scheduler = optim.lr_scheduler.ExponentialLR(
                        self.optimizer, self.decay)
                    self.unused = False
                print("Learning rate: {}".format(self.scheduler.get_lr()))

            # # exponential moving avarage
            # if self.use_ema and self.ema is not None:
            #     print("Apply ema")
            #     for name, param in self.model.named_parameters():
            #         if param.requires_grad:
            #             param.data = self.ema(name, param.data)

            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)

            # get prediction results and calculate metrics
            yp1 = torch.argmax(p1, 1)
            yp2 = torch.argmax(p2, 1)
            yps = torch.stack([yp1, yp2], dim=1)
            ymin, _ = torch.min(yps, 1)
            ymax, _ = torch.max(yps, 1)
            begin_, end_ = (ymin.float(),
                            ymax.float())
            begin, end = (y1.float(),
                          y2.float())

            em = em_by_begin_end_index(begin_, end_, begin, end)
            f1 = f1_by_begin_end_index(begin_, end_, begin, end)

            global_loss += loss.item()
            global_em += em
            global_f1 += f1

            # evaluate, log, and visualize for each print_freq batches
            if self.step % self.print_freq == self.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = self.train_data_loader.batch_size * \
                    step_num / used_time
                batch_loss = global_loss / step_num
                batch_em = global_em / step_num
                batch_f1 = global_f1 / step_num
                print("step %d / %d of epoch %d)" % (
                    batch_idx, len(self.train_data_loader), epoch),
                    file=self.logger, flush=True)
                print("loss: ", batch_loss, file=self.logger, flush=True)
                print("em: ", batch_em, file=self.logger, flush=True)
                print("f1: ", batch_f1, file=self.logger, flush=True)
                print("speed: %f examples/sec \n\n" % (speed),
                      file=self.logger, flush=True)
                if self.visualizer:
                    self.visualizer.plot('loss', global_loss / step_num)
                    self.visualizer.log(
                        "epoch:{epoch}, step:{step}, loss:{loss}, \
                        batch_em:{train_em}, batch_f1:{train_f1}".format(
                            epoch=epoch, step=self.step, loss=batch_loss,
                            batch_em=str(batch_em), batch_f1=str(batch_f1)))
                global_loss = 0.0
                global_em = 0.0
                global_f1 = 0.0
                last_step = self.step
                last_time = time.time()
            self.step += 1

            if self.is_debug and batch_idx >= self.debug_batchnum:
                break

        # evaluate, log, and visualize for each epoch
        train_em, train_f1 = self._valid_eopch(self.train_dataset,
                                               self.train_data_loader)
        dev_em, dev_f1 = self._valid_eopch(self.dev_dataset,
                                           self.dev_data_loader)
        print("train_em: %f" % train_em, file=self.logger, flush=True)
        print("train_f1: %f" % train_f1, file=self.logger, flush=True)
        print("dev_em: %f" % dev_em, file=self.logger, flush=True)
        print("dev_f1: %f" % dev_f1, file=self.logger, flush=True)
        if self.visualizer:
            self.visualizer.plot_many(
                {"train_em": train_em,
                 "train_f1": train_f1,
                 "dev_em": dev_em,
                 "dev_f1": dev_f1})
            self.visualizer.log(
                "epoch:{epoch}, \
                train_em:{train_em}, train_f1:{train_f1}, \
                dev_em:{dev_em}, dev_f1:{dev_f1}".format(
                    epoch=epoch,
                    train_em=str(train_em), train_f1=str(train_f1),
                    dev_em=str(dev_em), dev_f1=str(dev_f1)))

        result = {}
        result["em"] = dev_em
        result["f1"] = dev_f1
        return result

    def _valid_eopch(self, dataset, data_loader):
        """
        Evaluate model over development dataset.
        Return the metrics: em, f1.
        """
        self.model.eval()
        begin_ = []
        end_ = []
        begin = []
        end = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                (question_lengths,
                 question_wids,
                 question_cids,
                 context_lengths,
                 context_wids,
                 context_cids,
                 context_sent_lengths,
                 context_sent_wids,
                 context_sent_cids,
                 answer_spos,
                 answer_tpos,
                 answer_tpos_in_sent) = batch
                question_lengths = question_lengths.to(self.device)
                question_wids = question_wids.to(self.device)
                question_cids = question_cids.to(self.device)
                context_lengths = context_lengths.to(self.device)
                context_wids = context_wids.to(self.device)
                context_cids = context_cids.to(self.device)
                context_sent_lengths = context_sent_lengths.to(self.device)
                context_sent_wids = context_sent_wids.to(self.device)
                context_sent_cids = context_sent_cids.to(self.device)
                answer_spos = answer_spos.to(self.device)
                answer_tpos = answer_tpos.to(self.device)
                answer_tpos_in_sent = answer_tpos_in_sent.to(self.device)

                p1, p2 = self.model(
                    context_wids,
                    context_cids,
                    question_wids,
                    question_cids)
                y1, y2 = answer_tpos[:, 0], answer_tpos[:, 1]

                yp1 = torch.argmax(p1, 1)
                yp2 = torch.argmax(p2, 1)
                yps = torch.stack([yp1, yp2], dim=1)
                ymin, _ = torch.min(yps, 1)
                ymax, _ = torch.max(yps, 1)
                batch_begin_, batch_end_ = (ymin.float(), ymax.float())
                batch_begin, batch_end = (y1.float(), y2.float())

                begin_.extend(batch_begin_.cpu().numpy().tolist())
                end_.extend(batch_end_.cpu().numpy().tolist())
                begin.extend(batch_begin.cpu().numpy().tolist())
                end.extend(batch_end.cpu().numpy().tolist())

                if self.is_debug and batch_idx >= self.debug_batchnum:
                    break

        self.model.train()
        begin_ = torch.FloatTensor(begin_)
        end_ = torch.FloatTensor(end_)
        begin = torch.FloatTensor(begin)
        end = torch.FloatTensor(end)
        em = em_by_begin_end_index(begin_, end_, begin, end)
        f1 = f1_by_begin_end_index(begin_, end_, begin, end)
        return em, f1

    def _save_checkpoint(self, epoch, f1, em, is_best):
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_em': self.best_em,
            'step': self.step + 1,
            'start_time': self.start_time}
        if self.use_scheduler:
            state['unused'] = self.unused
        filename = os.path.join(
            self.save_dir,
            self.identifier +
            'checkpoint_epoch{:02d}_f1_{:.5f}_em_{:.5f}.pth.tar'.format(
                epoch, f1, em))
        print("Saving checkpoint: {} ...".format(filename),
              file=self.logger, flush=True)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(self.save_dir, 'model_best.pth.tar'))
        return filename

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path),
              file=self.logger, flush=True)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_f1 = checkpoint['best_f1']
        self.best_em = checkpoint['best_em']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.use_scheduler:
            self.scheduler.last_epoch = checkpoint['epoch']
            self.unused = checkpoint['unused']
        print("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch),
            file=self.logger, flush=True)
