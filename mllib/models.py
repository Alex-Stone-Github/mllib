import numpy as np

from . import bases

class Dense(bases.Module):
    """
    Simple dense fully-connected layer for linear regression
    """

    def __init__(self, inputNum, nodes):
        self.inputNum = inputNum
        self.nodes = nodes

        self.weights = np.random.rand(self.nodes, self.inputNum)
        self.biases = np.random.rand(self.nodes)
    
    def forward(self, inputs):
        """
        Forward pass method
        """
        outputs = np.empty(self.nodes)
        for i in range(self.nodes):
            outputs[i] = np.dot(inputs, self.weights[i])
        return outputs
    
    def backward(self, error, lr):
        """
        Compute gradients of the previous layer and adjust the weights of this layer
        """
        # compute gradients for the previous layer
        inputErrors = np.zeros(self.inputNum) # define gradients
        for nodeIndex in range(self.nodes):
            for inputIndex in range(self.inputNum):
                inputErrors[inputIndex] += error[nodeIndex] * self.weights[nodeIndex, inputIndex]

        # apply gradients to weights
        for i in range(self.nodes):
            self.weights[i] += error[i] * (np.abs(self.weights[i]) + .1) * lr
        
        return inputErrors


class Sigmoid(bases.PassBackModule):
    """
    Simple sigmoid activation function
    """
    
    def forward(self, inputs):
        return 1/(1+np.exp(-inputs))