import math
import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    """
    Add position information to input tensor.
    :Examples:
        >>> m = PositionalEncoding(d_model=6, max_len=10, dropout=0)
        >>> input = torch.randn(3, 10, 6)
        >>> output = m(input)
    """

    def __init__(self, d_model, dropout=0, max_len=5000):
        """
        :param d_model: same with input hidden size
        :param dropout: dropout rate
        :param max_len: maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :Input: (batch_num, seq_length, hidden_size)
        :Output: (batch_num, seq_length, hidden_size)
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
