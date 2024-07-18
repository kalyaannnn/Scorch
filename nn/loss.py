from .module import Module
from autograd.functions import *
from abc import ABC
from tensor import *

class Loss(Module, ABC):
    "Abstract Class for Loss Functions"

    def __init__(self):
        super().__init__
    
    def forward(self, predictions, labels):
        raise NotImplementedError
    
    def __call__(self, *inputs):
        return self.forward(*inputs)




class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, target):
        assert target.shape == predictions.shape, \
        "Labels and Predictions shape do not match : {} and {}".format(target.shape, predictions.shape)

        cost =  ((predictions - target) ** 2).sum() / predictions.numel
        return cost


