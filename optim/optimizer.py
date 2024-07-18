from abc import ABC
from scorch.tensor import Tensor
class Optimizer(ABC):
    "Abstract Class for Optimizer"

def __init__(self, parameters):
    if isinstance(parameters, Tensor):
        raise TypeError("Parameters should be iterable but got {} instead".format(type(parameters)))
    elif isinstance(parameters, dict):
        parameters = parameters.values()

    self.parameters = list(parameters)

def step(self):
    raise NotImplementedError

def zero_grad(self):
    for module, name, parameter in self.parameters:
        parameter.zero_grad()

class Adam(Optimizer):
    def __init__(self, parameters, lr = 1e-1, betas = (0.9, 0.999), eps = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0

        self._cache = {
            'm' : [p.zeros_like() for (_, _, p) in self.parameters],
            'v' : [p.zeros_like() for (_, _, p) in self.parameters]
        }

    def step(self):
        self.t += 1
        for i, (module, name, _) in enumerate(self.parameters):
            parameter = getattr(module, name)

            grad = parameter.grad
            m = self._cache['m'][i]
            v = self._cache['v'][i]

            m = self.betas[0] * m + (1 - self.betas[0]) * grad
            v = self.betas[1] * v + (1 - self.betas[1]) * grad * grad

            m_hat = m / (1 - self.betas[0] ** self.t)
            v_hat = v / (1 - self.betas[1] ** self.t)

            updated_parameter = parameter - self.lr * m_hat / (v_hat.sqrt() + self.eps)
            setattr(module, name, updated_parameter)

            self._cache['m'][i] = m
            self._cache['v'][i] = v

            parameter.detach()
            m.detach()
            v.detach()
    

            del parameter