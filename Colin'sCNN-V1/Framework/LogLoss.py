import numpy as np

class LogLoss:

    def eval(self, Y, Yhat):
        epsilon = 1e-15
        Yhat = np.clip(Yhat, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat), axis=1))

        return loss

    def gradient(self, Y, Yhat):
        epsilon = 1e-15
        Yhat = np.clip(Yhat, epsilon, 1 - epsilon)
        gradient = (1 - Y) / (1 - Yhat + epsilon) - Y / (Yhat + epsilon)

        return gradient
