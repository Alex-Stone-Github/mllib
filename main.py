import numpy as np
import mllib


class Model(mllib.bases.Module):
    def __init__(self):
        self.layer1 = mllib.models.Dense(10, 20)
        self.layer2 = mllib.models.Dense(20, 20)
        self.layer3 = mllib.models.Dense(20, 10)
        self.nonlin = mllib.models.Sigmoid()
    
    def forward(self, inputs):
        return self.nonlin(self.layer3(self.nonlin(self.layer2(self.nonlin(self.layer1(inputs)))))) # forward pass

    def backward(self, error, lr):
        error = self.nonlin.backward(error, lr)
        error = self.layer3.backward(error, lr)
        error = self.nonlin.backward(error, lr)
        error = self.layer2.backward(error, lr)
        error = self.nonlin.backward(error, lr)
        error = self.layer1.backward(error, lr)
        return error

model = Model()
x = np.random.randn(10)
y = np.random.randn(10)

def meanSquaredError(targets, values):
    return np.sum(np.abs(targets - values))


print(meanSquaredError(y, model(x)))
for _ in range(10):
    error = y - model(x)
    model.backward(error, .1)
print(meanSquaredError(y, model(x)))