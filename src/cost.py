import numpy as np

epsilon = 1e-20

class CostFunction:
    def f(self, a_last, y):
        raise NotImplementedError

    def grad(self, a_last, y):
        raise NotImplementedError
    
    def optimizedGrad(self, a_last, y):
        raise NotImplementedError

class SigmoidCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        # It would be better to have the logits and use this instead
        # max(logits, 0) - logits * y + log(1 + exp(-abs(logits)))
        a_last = np.clip(a_last, epsilon, 1.0 - epsilon)
        cost = -1 / batch_size * (y * np.log(a_last) + (1 - y) * np.log(1 - a_last)).sum()
        return cost

    # Only cross entropy derivative
    def grad(self, a_last, y):
        a_last = np.clip(a_last, epsilon, 1.0 - epsilon)
        return - (np.divide(y, a_last) - np.divide(1 - y, 1 - a_last))

    # Cross entropy derivative * sigmoid derivative
    def optimizedGrad(self, a_last, y):
        a_last = np.clip(a_last, epsilon, 1.0 - epsilon)
        return(a_last-y)


class SoftmaxCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        a_last = np.clip(a_last, epsilon, 1.0)
        cost = -1 / batch_size * (y * np.log(a_last)).sum()
        return cost
    
    # Only cross entropy derivative
    def grad(self, a_last, y):
        a_last = np.clip(a_last, epsilon, 1.0)
        return - np.divide(y, a_last)
    
    # Cross entropy derivative * softmax derivative
    def optimizedGrad(self, a_last, y):
        a_last = np.clip(a_last, epsilon, 1.0)
        return -(y-a_last)

softmax_cross_entropy = SoftmaxCrossEntropy()
sigmoid_cross_entropy = SigmoidCrossEntropy()