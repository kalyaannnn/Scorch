import math


class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]
    
    def backward(self, gradient):
        return [gradient, gradient]
    
class SinBackward:
    def __init__(self, x):
        self.input = [x]
    
    def backward(self, gradient):
        x = self.input[0]
        return [x.cos * gradient]
    
class cosBackward:
    def __init__(self, x):
        self.input = [x]
    
    def backward(self, gradient):
        x = self.input[0]
        return [- x.sin() * gradient]
    
class ElementWiseMulBackward:
    def __init__(self, x, y):
        self.input = [x, y]
    
    def backward(self, gradient):
        x = self.input[0]
        y = self.input[1]
        return [y * gradient, x * gradient]
    
class sumBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        return [float(gradient.tensor.contents.data[0]) * self.input[0].ones_like()]