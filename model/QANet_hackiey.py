"""
Reference: https://github.com/hackiey/QAnet-pytorch/tree/master/qanet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Highway(nn.Module):
    """ Version 1 : carry gate is (1 - transform gate)"""

    def __init__(self, input_size=500, n_layers=2):
        super(Highway, self).__init__()

        self.n_layers = n_layers

        self.transform = nn.ModuleList(
            [nn.Conv1d(input_size, input_size, kernel_size=1) for i in range(n_layers)])
        self.fc = nn.ModuleList([nn.Conv1d(input_size, input_size, kernel_size=1) for i in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            transformed = F.sigmoid(self.transform[i](x))
            carried = F.dropout(self.fc[i](x), p=0.1, training=self.training)
            x = carried * transformed + x * (1 - transformed)
            x = F.relu(x)

        return x


class WordEmbedding(nn.Module):
    def __init__(self, word_embeddings):
        super(WordEmbedding, self).__init__()

        self.word_embedding = nn.Embedding(num_embeddings=word_embeddings.shape[0],
                                           embedding_dim=word_embeddings.shape[1])

        self.word_embedding.weight = nn.Parameter(word_embeddings)
        self.word_embedding.weight.requires_grad = False

    def forward(self, input_context, input_question):
        context_word_emb = self.word_embedding(input_context)
        context_word_emb = F.dropout(context_word_emb, p=0.1, training=self.training)

        question_word_emb = self.word_embedding(input_question)
        question_word_emb = F.dropout(question_word_emb, p=0.1, training=self.training)

        return context_word_emb, question_word_emb


class CharacterEmbedding(nn.Module):
    def __init__(self, char_embeddings, embedding_dim=32, n_filters=200, kernel_size=5, padding=2):
        super(CharacterEmbedding, self).__init__()

        self.num_embeddings = len(char_embeddings)
        self.embedding_dim = embedding_dim
        self.kernel_size = (1, kernel_size)
        self.padding = (0, padding)

        self.char_embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=embedding_dim)
        self.char_embedding.weight = nn.Parameter(char_embeddings)
        # only difference is here: if we use_pretrained, random is different now.
        # as nn.Embedding used random weight, but from_pretrained doesn't use random
        self.char_conv = nn.Conv2d(in_channels=embedding_dim,
                                   out_channels=n_filters,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding)

    def forward(self, x):
        batch_size = x.shape[0]
        word_length = x.shape[-1]

        x = x.view(batch_size, -1)
        x = self.char_embedding(x)
        x = x.view(batch_size, -1, word_length, self.embedding_dim)

        # embedding dim of characters is number of channels of conv layer
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.char_conv(x))
        x = x.permute(0, 2, 3, 1)

        # max pooling over word length to have final tensor
        x, _ = torch.max(x, dim=2)

        x = F.dropout(x, p=0.05, training=self.training)

        return x


class InputEmbedding(nn.Module):
    def __init__(self, word_embeddings, char_embeddings, word_embed_dim=300,
                 char_embed_dim=32, char_embed_n_filters=200,
                 char_embed_kernel_size=7, char_embed_pad=3, highway_n_layers=2, hidden_size=128):

        super(InputEmbedding, self).__init__()

        self.word_embedding = WordEmbedding(word_embeddings)
        self.character_embedding = CharacterEmbedding(char_embeddings,
                                                     embedding_dim=char_embed_dim,
                                                     n_filters=char_embed_n_filters,
                                                     kernel_size=char_embed_kernel_size,
                                                     padding=char_embed_pad)

        self.projection = nn.Conv1d(word_embed_dim + char_embed_n_filters, hidden_size, 1)

        self.highway = Highway(input_size=hidden_size, n_layers=highway_n_layers)

    def forward(self, context_word, context_char, question_word, question_char):
        context_word, question_word = self.word_embedding(context_word, question_word)
        context_char = self.character_embedding(context_char)
        question_char = self.character_embedding(question_char)

        context = torch.cat((context_word, context_char), dim=-1)
        question = torch.cat((question_word, question_char), dim=-1)

        context = self.projection(context.permute(0, 2, 1))
        question = self.projection(question.permute(0, 2, 1))

        context = self.highway(context)
        question = self.highway(question)

        return context, question


class PositionEncoding(nn.Module):
    def __init__(self, n_filters=128, min_timescale=1.0, max_timescale=1.0e4):

        super(PositionEncoding, self).__init__()

        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.d = n_filters

        # we use the fact that cos(x) = sin(x + pi/2) to compute everything with one sin statement
        self.freqs = torch.Tensor(
            [max_timescale ** (-i / self.d) if i % 2 == 0 else max_timescale ** (-(i - 1) / self.d) for i in
             range(self.d)]).unsqueeze(1).to(device)
        self.phases = torch.Tensor([0 if i % 2 == 0 else np.pi / 2 for i in range(self.d)]).unsqueeze(1).to(device)

    def forward(self, x):

        # *************** speed up, static pos_enc ******************
        l = x.shape[-1]

        # computing signal
        pos = torch.arange(l).repeat(self.d, 1).to(device)
        tmp = pos * self.freqs + self.phases
        pos_enc = torch.sin(tmp)
        x = x + pos_enc

        return x

class LayerNorm1d(nn.Module):

    def __init__(self, n_features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return x.permute(0, 2, 1)



class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, n_filters=128, kernel_size=7, padding=3):
        super(DepthwiseSeparableConv1d, self).__init__()

        self.depthwise = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, groups=n_filters)
        self.separable = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.separable(x)

        return x


class SelfAttention(nn.Module):

    def __init__(self, n_heads=8, n_filters=128):
        super(SelfAttention, self).__init__()

        self.n_filters = n_filters
        self.n_heads = n_heads

        self.key_dim = n_filters // n_heads
        self.value_dim = n_filters // n_heads

        self.fc_query = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_key = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_value = nn.ModuleList([nn.Linear(n_filters, self.value_dim) for i in range(n_heads)])
        self.fc_out = nn.Linear(n_heads * self.value_dim, n_filters)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        l = x.shape[1]

        mask = mask.unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[1]).permute(0,2,1)

        heads = torch.zeros(self.n_heads, batch_size, l, self.value_dim, device=device)

        for i in range(self.n_heads):
            Q = self.fc_query[i](x)
            K = self.fc_key[i](x)
            V = self.fc_value[i](x)

            # scaled dot-product attention
            tmp = torch.bmm(Q, K.permute(0,2,1))
            tmp = tmp / np.sqrt(self.key_dim)
            tmp = F.softmax(tmp - 1e30*(1-mask), dim=-1)

            tmp = F.dropout(tmp, p=0.1, training=self.training)

            heads[i] = torch.bmm(tmp, V)

        # concatenation is the same as reshaping our tensor
        x = heads.permute(1,2,0,3).contiguous().view(batch_size, l, -1)
        x = self.fc_out(x)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, n_conv, kernel_size=7, padding=3, n_filters=128, n_heads=8, conv_type='depthwise_separable'):
        super(EncoderBlock, self).__init__()

        self.n_conv = n_conv
        self.n_filters = n_filters

        self.position_encoding = PositionEncoding(n_filters=n_filters)

        self.layer_norm = LayerNorm1d(n_features=n_filters)

        self.conv = nn.ModuleList([DepthwiseSeparableConv1d(n_filters,
                                                            kernel_size=kernel_size,
                                                            padding=padding) for i in range(n_conv)])
        self.self_attention = SelfAttention(n_heads, n_filters)

        self.fc = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            if torch.rand(1) > dropout:
                outputs = F.dropout(inputs, p=0.1, training=self.training)
                return outputs + residual
            else:
                return residual
        else:
            return inputs + residual

    def forward(self, x, mask, start_index, total_layers):

        outputs = self.position_encoding(x)

        # convolutional layers
        for i in range(self.n_conv):
            residual = outputs
            outputs = self.layer_norm(outputs)

            if i % 2 == 0:
                outputs = F.dropout(outputs, p=0.1, training=self.training)
            outputs = F.relu(self.conv[i](outputs))

            # layer dropout
            outputs = self.layer_dropout(outputs, residual, (0.1 * start_index / total_layers))
            start_index += 1

        # self attention
        residual = outputs
        outputs = self.layer_norm(outputs)

        outputs = F.dropout(outputs, p=0.1, training=self.training)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.self_attention(outputs, mask)
        outputs = outputs.permute(0, 2, 1)

        outputs = self.layer_dropout(outputs, residual, 0.1 * start_index / total_layers)
        start_index += 1

        # fully connected layer
        residual = outputs
        outputs = self.layer_norm(outputs)
        outputs = F.dropout(outputs, p=0.1, training=self.training)
        outputs = self.fc(outputs)
        outputs = self.layer_dropout(outputs, residual, 0.1 * start_index / total_layers)

        return outputs



class EmbeddingEncoder(nn.Module):
    def __init__(self, resize_in=500, hidden_size=128, resize_kernel=7, resize_pad=3,
                 n_blocks=1, n_conv=4, kernel_size=7, padding=3,
                 conv_type='depthwise_separable', n_heads=8, context_length=400, question_length=50):

        super(EmbeddingEncoder, self).__init__()

        self.n_conv = n_conv
        self.n_blocks = n_blocks
        self.total_layers = (n_conv+2)*n_blocks

        self.stacked_encoderBlocks = nn.ModuleList([EncoderBlock(n_conv=n_conv,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                n_filters=hidden_size,
                                                                conv_type=conv_type,
                                                                n_heads=n_heads) for i in range(n_blocks)])

    def forward(self, context_emb, question_emb, c_mask, q_mask):
        for i in range(self.n_blocks):
            context_emb = self.stacked_encoderBlocks[i](context_emb, c_mask, i*(self.n_conv+2)+1, self.total_layers)
            question_emb = self.stacked_encoderBlocks[i](question_emb, q_mask, i*(self.n_conv+2)+1, self.total_layers)

        return context_emb, question_emb

class ContextQueryAttention(nn.Module):

    def __init__(self, hidden_size=128):
        super(ContextQueryAttention, self).__init__()

        self.d = hidden_size

        self.W0 = nn.Linear(3 * self.d, 1)
        nn.init.xavier_normal_(self.W0.weight)

    def forward(self, C, Q, c_mask, q_mask):

        batch_size = C.shape[0]

        n = C.shape[2]
        m = Q.shape[2]

        q_mask.unsqueeze(-1)

        # Evaluate the Similarity matrix, S
        S = self.similarity(C.permute(0, 2, 1), Q.permute(0, 2, 1), n, m, batch_size)

        S_ = F.softmax(S - 1e30*(1-q_mask.unsqueeze(-1).permute(0, 2, 1).expand(batch_size, n, m)), dim=2)
        S__ = F.softmax(S - 1e30*(1-c_mask.unsqueeze(-1).expand(batch_size, n, m)), dim=1)

        A = torch.bmm(S_, Q.permute(0, 2, 1))
        #   AT = A.permute(0,2,1)
        B = torch.matmul(torch.bmm(S_, S__.permute(0, 2, 1)), C.permute(0, 2, 1))
        #   BT = B.permute(0,2,1)

        # following the paper, this layer should return the context2query attention
        # and the query2context attention
        return A, B

    def similarity(self, C, Q, n, m, batch_size):

        C = F.dropout(C, p=0.1, training=self.training)
        Q = F.dropout(Q, p=0.1, training=self.training)

        # Create QSim (#batch x n*m x d) where each of the m original rows are repeated n times
        Q_sim = self.repeat_rows_tensor(Q, n)
        # Create CSim (#batch x n*m x d) where C is reapted m times
        C_sim = C.repeat(1, m, 1)
        assert Q_sim.shape == C_sim.shape
        QC_sim = Q_sim * C_sim

        # The "learned" Similarity in 1 col, put back
        Sim_col = self.W0(torch.cat((Q_sim, C_sim, QC_sim), dim=2))
        # Put it back in right dim
        Sim = Sim_col.view(batch_size, m, n).permute(0, 2, 1)

        return Sim

    def repeat_rows_tensor(self, X, rep):
        (depth, _, col) = X.shape
        # Open dim after batch ("depth")
        X = torch.unsqueeze(X, 1)
        # Repeat the matrix in the dim opened ("depth")
        X = X.repeat(1, rep, 1, 1)
        # Permute depth and lines to get the repeat over lines
        X = X.permute(0, 2, 1, 3)
        # Return to input (#batch x #lines*#repeat x #cols)
        X = X.contiguous().view(depth, -1, col)

        return X



class ModelEncoder(nn.Module):
    def __init__(self, n_blocks=7, n_conv=2, kernel_size=7, padding=3,
                 hidden_size=128, conv_type='depthwise_separable', n_heads=8, context_length=400):
        
        super(ModelEncoder, self).__init__()

        self.n_conv = n_conv
        self.n_blocks = n_blocks
        self.total_layers = (n_conv + 2) * n_blocks

        self.stacked_encoderBlocks = nn.ModuleList([EncoderBlock(n_conv=n_conv,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                n_filters=hidden_size,
                                                                conv_type=conv_type,
                                                                n_heads=n_heads) for i in range(n_blocks)])

    def forward(self, x, mask):
        
        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask, i*(self.n_conv+2)+1, self.total_layers)
        M0 = x

        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask, i*(self.n_conv+2)+1, self.total_layers)
        M1 = x

        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask, i*(self.n_conv+2)+1, self.total_layers)

        M2 = x

        return M0, M1, M2


class Output(nn.Module):
    def __init__(self, input_dim = 512):
        super(Output, self).__init__()

        self.d = input_dim

        self.W1 = nn.Linear(2*self.d, 1)
        self.W2 = nn.Linear(2*self.d, 1)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, M0, M1, M2):

        p1 = self.W1(torch.cat((M0,M1), -1)).squeeze()
        p2 = self.W2(torch.cat((M0,M2), -1)).squeeze()
        return p1, p2



class QANet(nn.Module):
    ''' All-in-one wrapper for all modules '''

    def __init__(self, word_embeddings, char_embeddings,
                 c_max_len, q_max_len, d_model, train_cemb=False, pad=0,
                 dropout=0.1, num_head=1):
        super(QANet, self).__init__()
        self.PAD = pad

        params = {
            "n_epochs": 30,
            "batch_size":32,
            "learning_rate": 1e-3,
            "beta1": 0.8,
            "beta2": 0.999,
            "weight_decay": 3e-7,

            "word_embed_dim": 300,

            "char_dim": 64,
            "char_embed_n_filters": 128,
            "char_embed_kernel_size": 5,
            "char_embed_pad": 2,

            "highway_n_layers": 2,

            "hidden_size": 128,

            "embed_encoder_resize_kernel_size": 5,
            "embed_encoder_resize_pad": 3,

            "embed_encoder_n_blocks": 1,
            "embed_encoder_n_conv": 4,
            "embed_encoder_kernel_size": 7,
            "embed_encoder_pad": 3,
            "embed_encoder_conv_type": "depthwise_separable",
            "embed_encoder_with_self_attn": False,
            "embed_encoder_n_heads": 1,

            "model_encoder_n_blocks": 7,
            "model_encoder_n_conv": 2,
            "model_encoder_kernel_size": 5,
            "model_encoder_pad": 2,
            "model_encoder_conv_type": "depthwise_separable",
            "model_encoder_with_self_attn": False,
            "model_encoder_n_heads": 1,

            "para_limit": 400,
            "ques_limit": 50,
            "ans_limit": 30,
            "char_limit": 16
        }

        self.batch_size = params['batch_size']

        # Defining dimensions using data from the params.json file
        self.word_embed_dim = params['word_embed_dim']
        self.char_word_len = params["char_limit"]

        self.context_length = c_max_len
        self.question_length = q_max_len

        self.char_embed_dim = params["char_dim"]
        self.char_embed_n_filters = params["char_embed_n_filters"]
        self.char_embed_kernel_size = params["char_embed_kernel_size"]
        self.char_embed_pad = params["char_embed_pad"]

        self.highway_n_layers = params["highway_n_layers"]

        self.hidden_size = params["hidden_size"]

        self.embed_encoder_resize_kernel_size = params["embed_encoder_resize_kernel_size"]
        self.embed_encoder_resize_pad = params["embed_encoder_resize_pad"]

        self.embed_encoder_n_blocks = params["embed_encoder_n_blocks"]
        self.embed_encoder_n_conv = params["embed_encoder_n_conv"]
        self.embed_encoder_kernel_size = params["embed_encoder_kernel_size"]
        self.embed_encoder_pad = params["embed_encoder_pad"]
        self.embed_encoder_conv_type = params["embed_encoder_conv_type"]
        self.embed_encoder_with_self_attn = params["embed_encoder_with_self_attn"]
        self.embed_encoder_n_heads = params["embed_encoder_n_heads"]

        self.model_encoder_n_blocks = params["model_encoder_n_blocks"]
        self.model_encoder_n_conv = params["model_encoder_n_conv"]
        self.model_encoder_kernel_size = params["model_encoder_kernel_size"]
        self.model_encoder_pad = params["model_encoder_pad"]
        self.model_encoder_conv_type = params["model_encoder_conv_type"]
        self.model_encoder_with_self_attn = params["model_encoder_with_self_attn"]
        self.model_encoder_n_heads = params["model_encoder_n_heads"]

        # Initializing model layers
        # word_embeddings = np.array(word_embeddings)
        self.input_embedding = InputEmbedding(word_embeddings,
                                             char_embeddings,
                                             word_embed_dim=self.word_embed_dim,
                                             char_embed_dim=self.char_embed_dim,
                                             char_embed_n_filters=self.char_embed_n_filters,
                                             char_embed_kernel_size=self.char_embed_kernel_size,
                                             char_embed_pad=self.char_embed_pad,
                                             highway_n_layers=self.highway_n_layers,
                                             hidden_size=self.hidden_size)

        self.embedding_encoder = EmbeddingEncoder(resize_in=self.word_embed_dim + self.char_embed_n_filters,
                                                 hidden_size=self.hidden_size,
                                                 resize_kernel=self.embed_encoder_resize_kernel_size,
                                                 resize_pad=self.embed_encoder_resize_pad,
                                                 n_blocks=self.embed_encoder_n_blocks,
                                                 n_conv=self.embed_encoder_n_conv,
                                                 kernel_size=self.embed_encoder_kernel_size,
                                                 padding=self.embed_encoder_pad,
                                                 conv_type=self.embed_encoder_conv_type,
                                                 n_heads=self.embed_encoder_n_heads,
                                                 context_length=self.context_length,
                                                 question_length=self.question_length)

        self.context_query_attention = ContextQueryAttention(hidden_size=self.hidden_size)

        self.projection = nn.Conv1d(4 * self.hidden_size, self.hidden_size, kernel_size=1)

        self.model_encoder = ModelEncoder(n_blocks=self.model_encoder_n_blocks,
                                         n_conv=self.model_encoder_n_conv,
                                         kernel_size=self.model_encoder_kernel_size,
                                         padding=self.model_encoder_pad,
                                         hidden_size=self.hidden_size,
                                         conv_type=self.model_encoder_conv_type,
                                         n_heads=self.model_encoder_n_heads)
        self.output = Output(input_dim=self.hidden_size)

    def forward(self, context_word, context_char, question_word, question_char):
        c_mask = (torch.ones_like(context_word) *
                 self.PAD != context_word).float()
        q_mask = (torch.ones_like(question_word) *
                 self.PAD != question_word).float()

        c_maxlen = int(c_mask.sum(1).max().item())
        q_maxlen = int(q_mask.sum(1).max().item())
        context_word = context_word[:, :c_maxlen]
        context_char = context_char[:, :c_maxlen, :]
        question_word = question_word[:, :q_maxlen]
        question_char = question_char[:, :q_maxlen, :]
        c_mask = c_mask[:, :c_maxlen]
        q_mask = q_mask[:, :q_maxlen]

        context_emb, question_emb = self.input_embedding(context_word, context_char, question_word, question_char)
        context_emb, question_emb = self.embedding_encoder(context_emb, question_emb, c_mask, q_mask)

        c2q_attn, q2c_attn = self.context_query_attention(context_emb, question_emb, c_mask, q_mask)
        mdl_emb = torch.cat((context_emb,
                  c2q_attn.permute(0, 2, 1),
                  context_emb*c2q_attn.permute(0, 2, 1),
                  context_emb*q2c_attn.permute(0, 2, 1)), 1)

        mdl_emb = self.projection(mdl_emb)
        M0, M1, M2 = self.model_encoder(mdl_emb, c_mask)

        p1, p2 = self.output(M0.permute(0,2,1), M1.permute(0,2,1), M2.permute(0,2,1))

        p1 = p1 - 1e30*(1 - c_mask)
        p2 = p2 - 1e30*(1 - c_mask)

        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)
