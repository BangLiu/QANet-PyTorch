"""
Exponential Moving Average for model parameters.

Sample usage:
    # initialize
    ema = EMA(0.999)

    # register model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    # in batch training loop
    # for batch in batches:
    optimizer.step()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = ema(name, param.data)
"""


class EMA(object):

    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
