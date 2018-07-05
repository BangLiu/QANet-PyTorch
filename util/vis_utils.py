# -*- coding: utf-8 -*-
from torchviz import make_dot
from graphviz import Source


def plot_model(model, output, fig_name, fig_format):
    """
    Plot neural network model architecture
    :param model: network model
    :param output: network output variable
    :param fig_name: figure output name
    :param format: figure format, such as "png", "pdf"
    :return: dot file string
    """
    var = make_dot(output, params=dict(model.named_parameters()))
    s = Source(var, filename=fig_name, format=fig_format)
    s.view()
    return var
