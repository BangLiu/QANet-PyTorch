# -*- coding: utf-8 -*-
"""
Main model architecture.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.cnn import DepthwiseSeparableConv
from .modules.attention import MultiHeadAttention
from .modules.position import PositionalEncoding
from .modules.highway import Highway


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class Embedding(nn.Module):

    def __init__(self, wemb_dim, cemb_dim, d_model,
                 dropout_w=0.1, dropout_c=0.05):
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(
            cemb_dim, d_model, 5, dim=2, bias=True)
        self.conv1d = DepthwiseSeparableConv(
            wemb_dim + d_model, d_model, 5, bias=True)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb.transpose(1, 2))
        return emb


class EncoderBlock(nn.Module):

    def __init__(self, conv_num, d_model, h, k, length, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [DepthwiseSeparableConv(d_model, d_model, k)
             for _ in range(conv_num)])
        self.self_att = MultiHeadAttention(h=h, d_model=d_model)
        self.fc = nn.Linear(d_model, d_model, bias=True)
        self.pos = PositionalEncoding(d_model)
        self.normb = nn.LayerNorm([length, d_model])
        self.norms = nn.ModuleList(
            [nn.LayerNorm([length, d_model])
             for _ in range(conv_num)])
        self.norme = nn.LayerNorm([length, d_model])
        self.dropout = dropout
        self.L = conv_num

    def forward(self, x, mask):
        out = self.pos(x)
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out.transpose(1, 2)).transpose(1, 2)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        out, _ = self.self_att(out, out, out, mask)
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class CQAttention(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w = torch.empty(d_model * 3)
        lim = 1 / d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.dropout = dropout
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        batch_size, c_max_len, d_model = C.shape
        batch_size, q_max_len, d_model = Q.shape
        ss = []
        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)
        for i in range(q_max_len):
            q = Q[:, i, :].unsqueeze(1)
            QCi = torch.mul(q, C)
            Qi = q.expand(batch_size, c_max_len, d_model)
            Xi = torch.cat([Qi, C, QCi], dim=2)
            Si = torch.matmul(Xi, self.w).unsqueeze(2)
            ss.append(Si)
        S = torch.cat(ss, dim=2)
        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class Pointer(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        w1 = torch.rand(d_model * 2)
        w2 = torch.rand(d_model * 2)
        lim = 3 / (2 * d_model)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1.transpose(1, 2), M2.transpose(1, 2)], dim=1)
        X2 = torch.cat([M1.transpose(1, 2), M3.transpose(1, 2)], dim=1)
        Y1 = torch.matmul(self.w1, X1)
        Y2 = torch.matmul(self.w2, X2)
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2


class QANet(nn.Module):

    def __init__(self, wv_tensor, cv_tensor,
                 c_max_len, q_max_len, d_model, train_cemb=False, pad=0):
        super().__init__()
        if train_cemb:
            self.char_emb = nn.Embedding.from_pretrained(
                cv_tensor, freeze=False)
        else:
            self.char_emb = nn.Embedding.from_pretrained(cv_tensor)
        self.word_emb = nn.Embedding.from_pretrained(wv_tensor)
        wemb_dim = wv_tensor.shape[1]
        cemb_dim = cv_tensor.shape[1]
        self.emb = Embedding(wemb_dim, cemb_dim, d_model)

        self.c_emb_enc = EncoderBlock(
            conv_num=4, d_model=d_model, h=8, k=7, length=c_max_len)
        self.q_emb_enc = EncoderBlock(
            conv_num=4, d_model=d_model, h=8, k=7, length=q_max_len)
        self.cq_att = CQAttention(d_model=d_model)

        self.cq_resizer = DepthwiseSeparableConv(d_model * 4, d_model, 5)
        enc_blk = EncoderBlock(
            conv_num=2, d_model=d_model, h=8, k=5, length=c_max_len)
        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)

        self.out = Pointer(d_model)
        self.PAD = pad

    def forward(self, context_wids, context_cids,
                question_wids, question_cids):
        cmask = (torch.ones_like(context_wids) *
                 self.PAD != context_wids).float()
        qmask = (torch.ones_like(question_wids) *
                 self.PAD != question_wids).float()
        Cw, Cc = self.word_emb(context_wids), self.char_emb(context_cids)
        Qw, Qc = self.word_emb(question_wids), self.char_emb(question_cids)
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)

        ccmask = torch.stack([cmask] * cmask.shape[1], dim=1) * \
            torch.stack([cmask] * cmask.shape[1], dim=2)
        qqmask = torch.stack([qmask] * qmask.shape[1], dim=1) * \
            torch.stack([qmask] * qmask.shape[1], dim=2)
        Ce = self.c_emb_enc(C, ccmask)
        Qe = self.q_emb_enc(Q, qqmask)
        X = self.cq_att(Ce, Qe, cmask, qmask)

        M1 = self.cq_resizer(X.transpose(1, 2)).transpose(1, 2)
        for enc in self.model_enc_blks:
            M1 = enc(M1, ccmask)
        M2 = M1
        for enc in self.model_enc_blks:
            M2 = enc(M2, ccmask)
        M3 = M2
        for enc in self.model_enc_blks:
            M3 = enc(M3, ccmask)
        p1, p2 = self.out(M1, M2, M3, cmask)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)


if __name__ == "__main__":
    test_EncoderBlock = False
    test_QANet = True
    test_QANet_separate = False
    test_QANet_whole = True

    if test_EncoderBlock:
        batch_size = 32
        seq_length = 20
        hidden_dim = 128
        x = torch.rand(batch_size, seq_length, hidden_dim)
        m = EncoderBlock(4, hidden_dim, 8, 7, seq_length)
        y = m(x, mask=None)

    if test_QANet:
        # device and data sizes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wemb_vocab_size = 5000
        wemb_dim = 300
        cemb_vocab_size = 94
        cemb_dim = 64
        d_model = 128
        batch_size = 32
        q_max_len = 50
        c_max_len = 400
        char_dim = 16

        # fake embedding
        wv_tensor = torch.rand(wemb_vocab_size, wemb_dim)
        cv_tensor = torch.rand(cemb_vocab_size, cemb_dim)

        # fake input
        question_lengths = torch.LongTensor(batch_size).random_(1, q_max_len)
        question_wids = torch.zeros(batch_size, q_max_len).long()
        question_cids = torch.zeros(batch_size, q_max_len, char_dim).long()
        context_lengths = torch.LongTensor(batch_size).random_(1, c_max_len)
        context_wids = torch.zeros(batch_size, c_max_len).long()
        context_cids = torch.zeros(batch_size, c_max_len, char_dim).long()
        for i in range(batch_size):
            question_wids[i, 0:question_lengths[i]] = \
                torch.LongTensor(1, question_lengths[i]).random_(
                    1, wemb_vocab_size)
            question_cids[i, 0:question_lengths[i], :] = \
                torch.LongTensor(1, question_lengths[i], char_dim).random_(
                    1, cemb_vocab_size)
            context_wids[i, 0:context_lengths[i]] = \
                torch.LongTensor(1, context_lengths[i]).random_(
                    1, wemb_vocab_size)
            context_cids[i, 0:context_lengths[i], :] = \
                torch.LongTensor(1, context_lengths[i], char_dim).random_(
                    1, cemb_vocab_size)

        if test_QANet_separate:
            # test qanet all layers
            # embedding
            char_emb = nn.Embedding.from_pretrained(cv_tensor)
            word_emb = nn.Embedding.from_pretrained(wv_tensor)
            emb = Embedding(wemb_dim, cemb_dim, d_model)
            cmask = (torch.zeros_like(context_wids) != context_wids).float()
            qmask = (torch.zeros_like(question_wids) != question_wids).float()
            Cw, Cc = word_emb(context_wids), char_emb(context_cids)
            Qw, Qc = word_emb(question_wids), char_emb(question_cids)
            C = emb(Cc, Cw)
            Q = emb(Qc, Qw)

            # encoding
            c_emb_enc = EncoderBlock(
                conv_num=4, d_model=d_model, h=8, k=7, length=c_max_len)
            q_emb_enc = EncoderBlock(
                conv_num=4, d_model=d_model, h=8, k=7, length=q_max_len)
            cq_att = CQAttention(d_model=d_model)
            ccmask = torch.stack([cmask] * cmask.shape[1], dim=1) * \
                torch.stack([cmask] * cmask.shape[1], dim=2)
            Ce = c_emb_enc(C, ccmask)
            qqmask = torch.stack([qmask] * qmask.shape[1], dim=1) * \
                torch.stack([qmask] * qmask.shape[1], dim=2)
            Qe = q_emb_enc(Q, qqmask)
            X = cq_att(Ce, Qe, cmask, qmask)

            # attention
            cq_resizer = DepthwiseSeparableConv(d_model * 4, d_model, 5)
            enc_blk = EncoderBlock(
                conv_num=2, d_model=d_model, h=8, k=5, length=c_max_len)
            model_enc_blks = nn.ModuleList([enc_blk] * 7)
            out = Pointer(d_model)
            M1 = cq_resizer(X.transpose(1, 2)).transpose(1, 2)
            for enc in model_enc_blks:
                M1 = enc(M1, ccmask)
            M2 = M1
            for enc in model_enc_blks:
                M2 = enc(M2, ccmask)
            M3 = M2
            for enc in model_enc_blks:
                M3 = enc(M3, ccmask)

            # pointer
            p1, p2 = out(M1, M2, M3, cmask)
            print(p1)
            print(p2)

        # test whole QANet
        if test_QANet_whole:
            qanet = QANet(wv_tensor, cv_tensor,
                          c_max_len, q_max_len, d_model, train_cemb=False)
            p1, p2 = qanet(context_wids, context_cids,
                           question_wids, question_cids)
            print(p1.shape)
            print(p2.shape)
