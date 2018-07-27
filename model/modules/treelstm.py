"""
Define the Tree LSTM model in paper:
"Learning to Compose Task-Specific Tree Structures"
"""
import torch
from torch import nn
from torch.nn import init
from . import treelstm_utils


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, left=None, right=None):
        """
        :param left: A (h_l, c_l) tuple, where each value has the size
                     (batch_size, max_length, hidden_dim).
        :param right: A (h_r, c_r) tuple, where each value has the size
                      (batch_size, max_length, hidden_dim).
        :returns: h, c, The hidden and cell state of the composed parent,
                  each of which has the size
                  (batch_size, max_length, hidden_dim).
        """
        hl, cl = left
        hr, cr = right
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = treelstm_utils.apply_nd(
            fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() +
             cr * (fr + 1).sigmoid() +
             u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class BinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, intra_attention,
                 gumbel_temperature, bidirectional):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.gumbel_temperature = gumbel_temperature
        self.bidirectional = bidirectional

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(  # _bw means backward
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        if self.bidirectional:
            self.treelstm_layer = BinaryTreeLSTMLayer(2 * hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * hidden_dim))
        else:
            self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal_(self.word_linear.weight.data)
            init.constant_(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        init.normal_(self.comp_query.data, mean=0, std=0.01)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().\
            unsqueeze(1).unsqueeze(2).expand_as(new_h)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = treelstm_utils.dot_nd(
            query=self.comp_query, candidates=new_h)
        if self.training:
            select_mask = treelstm_utils.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature,
                mask=mask)
        else:
            select_mask = treelstm_utils.greedy_select(
                logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = select_mask_cumsum.data.new(
            new_h.size(0), 1).zero_()
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h +
                 left_mask_expand * old_h_left +
                 right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c +
                 left_mask_expand * old_c_left +
                 right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, length, return_select_masks=True):
        """
        :param input: (Tensor) a batch of text representations, with size
                      (batch_size, max_length, hidden_dim).
        :param length: (Tensor) a vector of length batch_size that records
                       each sample's length in the input batch.
        :param return_select_masks: whether return the inter-media select masks
                                    that record how the tree is composed.
        :returns: h, (batch_size, hidden_dim), the final hidden states.
                  c, (batch_size, hidden_dim), the final cell states.
                  nodes, (batch_size, 2 * max_length - 1, hidden_dim),
                  all the hidden states during the composition.
                  select_masks, record how the trees are composed.
                  boundaries, (batch_size, 2 * max_length - 1, 2)
                  It is calculated from select_masks.
                  It record each node's cover boundary in the original
                  sentence, from bottom to top root node.
        """
        batch_size, max_length, _ = input.size()
        max_depth = input.size(1)
        length_mask = treelstm_utils.sequence_mask(seq_length=length,
                                                   max_length=max_depth)
        select_masks = []

        # STEP 1: calculate leaf nodes' h and c.
        # If we use_leaf_rnn, it means we calculate leaf's (or words') h, c
        # by an LSTM or BiLSTM. If we do not use_leaf_rnn, then we just use
        # a linear layer to get leaf's h, c.
        # The output state = (h, c), where h and c size are both
        # (batch_size, max_length, hidden_dim * num_direction).
        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = input.data.new(batch_size, self.hidden_dim).zero_()
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = treelstm_utils.reverse_padded_sequence(
                    inputs=input, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = treelstm_utils.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = treelstm_utils.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = treelstm_utils.apply_nd(fn=self.word_linear, input=input)
            state = state.chunk(chunks=2, dim=2)

        # STEP 2: calculate inter media nodes' h and c,
        # and record combine position by select_masks
        nodes = []
        nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            new_state = self.treelstm_layer(left=(h[:, :-1, :], c[:, :-1, :]),
                                            right=(h[:, 1:, :], c[:, 1:, :]))
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = \
                    self.select_composition(
                        old_state=state, new_state=new_state,
                        mask=length_mask[:, i + 1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                nodes.append(selected_h.unsqueeze_(1))
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state  # sizes are (batch_size, 1, hidden_dim * num_direction)

        # If we use intra_attention, then the final h will be an
        # attention-weighted aggregation of all nodes' h.
        if self.intra_attention:
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask = att_mask.float()
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            nodes = torch.cat(nodes, dim=1)
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            nodes = nodes * att_mask_expand
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = treelstm_utils.masked_softmax(
                logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1).unsqueeze(1)
        assert h.size(1) == 1 and c.size(1) == 1

        median_boundaries = self._select_masks_to_boundaries(select_masks)
        leaf_boundaries = torch.rand(batch_size, max_length, 2).long()
        for j in range(max_length):
            leaf_boundaries[:, j, :] = j
        root_boundaries = torch.rand(batch_size, 1, 2).long()
        root_boundaries[:, :, 1] = max_length - 1
        boundaries = torch.cat([leaf_boundaries,
                                median_boundaries,
                                root_boundaries],
                               dim=1)

        if not return_select_masks:
            # result sizes are both (batch_size, hidden_dim)
            return h.squeeze(1), c.squeeze(1)
        else:
            # size of nodes are (batch_size, 2 * max_length - 1, hidden_dim)
            return h.squeeze(1), c.squeeze(1), select_masks, nodes, boundaries

    def _select_masks_to_boundaries(self, select_masks):
        """
        Transform select_masks to boundaries.
        :param select_masks: a list of select_mask, shapes are
                             [(batch_size, max_length - 1),
                              (batch_size, max_length - 2),
                              ...,
                              (batch_size, 2)]
        :return: inter-media nodes boundaries,
                 shape is (batch_size, max_length - 2, 2),
                 each element in last dimension is [start, end] index.
        """
        def _merge(node_covers, select_idx):
            """
            node_covers [[s1, e1], [s2, e2], ..., [sn, en]]
            """
            new_node_covers = []
            for i in range(len(node_covers) - 1):
                if i == select_idx:
                    merge_node_boundary = [node_covers[select_idx][0],
                                           node_covers[select_idx + 1][1]]
                    new_node_covers.append(merge_node_boundary)
                elif i < select_idx:
                    new_node_covers.append(node_covers[i])
                else:
                    new_node_covers.append(node_covers[i + 1])
            return new_node_covers, merge_node_boundary

        batch_size = select_masks[0].size()[0]
        max_length = select_masks[0].size()[1] + 1
        combine_matrix = torch.rand(batch_size, max_length, 2).long()
        for j in range(max_length):
            combine_matrix[:, j, :] = j

        results = []
        for batch_idx in range(batch_size):
            node_covers = combine_matrix[batch_idx, :, :].numpy().tolist()
            result = []
            for node_idx in range(max_length - 2):
                select = select_masks[node_idx][batch_idx, :]
                select_idx = torch.nonzero(select).data[0][0]
                node_covers, merge_boundary = _merge(node_covers, select_idx)
                result.append(merge_boundary)
            results.append(result)
        results = torch.LongTensor(results)
        return results


if __name__ == "__main__":
    # test BinaryTreeLSTMLayer
    hidden_dim = 5
    batch_size = 3
    max_length = 6
    hl = torch.randn(batch_size, max_length, hidden_dim)
    cl = torch.randn(batch_size, max_length, hidden_dim)
    left = tuple([hl, cl])
    hr = torch.randn(batch_size, max_length, hidden_dim)
    cr = torch.randn(batch_size, max_length, hidden_dim)
    right = tuple([hr, cr])
    model = BinaryTreeLSTMLayer(hidden_dim)
    h, c = model.forward(left, right)
    print(h.shape)
    print(c.shape)

    # test BinaryTreeLSTM
    word_dim = 3
    hidden_dim = 3
    use_leaf_rnn = True
    intra_attention = True
    gumbel_temperature = 1
    bidirectional = True
    model = BinaryTreeLSTM(word_dim, hidden_dim, use_leaf_rnn, intra_attention,
                           gumbel_temperature, bidirectional)
    batch_size = 2
    max_length = 6
    input = torch.randn(batch_size, max_length, hidden_dim)
    length = torch.LongTensor([6, 5])
    h, c, select_masks, nodes, boundaries = model.forward(input, length, True)
    print(h.size())
    print(c.size())
    print(nodes.size())
    print(select_masks)
    print(boundaries)
