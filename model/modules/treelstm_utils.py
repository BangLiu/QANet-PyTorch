"""
Basic or helper implementation.
"""
import torch
from torch.nn import functional


def apply_nd(fn, input):
    """
    Apply fn whose output only depends on the last dimension values
    to an arbitrary n-dimensional input.
    It flattens dimensions except the last one, applies fn, and then
    restores the original size.
    """
    x_size = input.size()
    x_flat = input.view(-1, x_size[-1])
    output_flat = fn(x_flat)
    output_size = x_size[:-1] + (output_flat.size(-1),)
    return output_flat.view(*output_size)


def affine_nd(input, weight, bias):
    """
    An helper function to make applying the "wx + b" operation for
    n-dimensional x easier.
    :param input: (Tensor) An arbitrary input data, whose size is
                  (d0, d1, ..., dn, input_dim)
    :param weight: (Tensor) A matrix of size (output_dim, input_dim)
    :param bias: (Tensor) A bias vector of size (output_dim,)
    :returns: The result of size (d0, ..., dn, output_dim)
    """
    input_size = input.size()
    input_flat = input.view(-1, input_size[-1])
    bias_expand = bias.unsqueeze(0).expand(input_flat.size(0), bias.size(0))
    output_flat = torch.addmm(bias_expand, input_flat, weight)
    output_size = input_size[:-1] + (weight.size(1),)
    output = output_flat.view(*output_size)
    return output


def dot_nd(query, candidates):
    """
    Perform a dot product between a query and n-dimensional candidates.
    :param query: (Tensor) A vector to query, whose size is
                  (query_dim,)
    :param candidates: (Tensor) A n-dimensional tensor to be multiplied
                       by query, whose size is (d0, d1, ..., dn, query_dim)
    :returns: The result of the dot product, whose size is
              (d0, d1, ..., dn)
    """

    cands_size = candidates.size()
    cands_flat = candidates.view(-1, cands_size[-1])
    output_flat = torch.mv(cands_flat, query)
    output = output_flat.view(*cands_size[:-1])
    return output


def convert_to_one_hot(indices, num_classes):
    """
    :param indices: (Tensor) A vector containing indices,
                    whose size is (batch_size,).
    :param num_classes: (Tensor) The number of classes, which would be
                        the second dimension of the resulting one-hot matrix.
    :returns: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.data.new(
        batch_size, num_classes).zero_().scatter_(1, indices.data, 1)
    return one_hot


def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = functional.softmax(logits, dim=1)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot


def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.
    :param logits: (Tensor) A un-normalized probability values,
                   which has the size (batch_size, num_classes)
    :param temperature: (float) A temperature parameter. The higher
                        the value is, the smoother the distribution is.
    :param mask: (Tensor, optional) If given, it masks the softmax
                 so that indices of '0' mask values are not selected.
                 The size is (batch_size, num_classes).
    :returns: The sampled output, which has the property explained above.
    """
    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(
        indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y


def sequence_mask(seq_length, max_length=None):
    if max_length is None:
        max_length = seq_length.data.max()
    batch_size = seq_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    if seq_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = seq_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """
    Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    :param inputs: (Tensor) padded batch of variable length sequences.
    :param lengths: (list[int]) list of sequence lengths
    :param batch_first: (bool, optional) if True, inputs should be B x T x *.
    :returns: A Tensor with the same size as inputs, but with each sequence
              reversed according to its length.
    """
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length - 1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    if inputs.is_cuda:
        device = inputs.get_device()
        reversed_indices = reversed_indices.cuda(device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


if __name__ == "__main__":
    # test reverse_padded_sequence
    print("reverse_padded_sequence")
    inputs = torch.LongTensor([[[1], [2], [3], [0]],
                               [[4], [4], [0], [0]],
                               [[3], [5], [6], [8]]])
    lengths = [3, 2, 4]
    batch_first = True
    result = reverse_padded_sequence(inputs, lengths, batch_first)
    print(result)

    # test masked_softmax
    print("masked_softmax")
    logits = torch.FloatTensor([[1, 2], [3, 2], [1, 5]])
    mask = torch.LongTensor([[1, 1], [1, 0], [0, 0]])
    result = masked_softmax(logits, mask)
    print(result)

    # test sequence_mask
    print("sequence_mask")
    seq_length = torch.LongTensor([2, 3, 5, 4, 1])
    max_length = 4
    print(sequence_mask(seq_length, max_length))

    # test st_gumbel_softmax
    print("st_gumbel_softmax")
    logits = torch.FloatTensor([[1, 2, 5, 2, 3],
                                [3, 2, 8, 1, 1],
                                [1, 5, 9, 3, 1]])
    print(st_gumbel_softmax(logits, temperature=1.0, mask=None))

    # test greedy_select
    print("greedy_select")
    logits = torch.FloatTensor([[10, 2, 5, 2, 3],
                                [3, 2, 8, 1, 11],
                                [1, 5, 9, 30, 1]])
    print(greedy_select(logits, mask=None))
