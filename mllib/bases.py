class Module:
    """
    Simple interface for model creation
    """
    def forward(self, inputs):
        raise NotImplemented("The forward method is not implemented for this class")
    def backward(self, error, lr):
        raise NotImplemented("The backward method is not implemented for this class")
    def __call__(self, inputs):
        return self.forward(inputs)

class PassBackModule(Module):
    """
    Simple way for model creation that preimplements the backward method
    """
    def backward(self, error, lr):
        return error