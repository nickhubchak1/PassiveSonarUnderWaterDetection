import numpy as np
from .Layer import Layer

class AddLayer(Layer):
    def __init__(self):
        """Initializes the AddLayer, which acts as a residual addition layer."""
        super().__init__()
        self.skip_X = None  # Stores residual input

    def forward(self, X, skip_X=None):
        """
        Performs the forward pass by adding the skip connection to the input.
        X: Current input from the main network path.
        skip_X: Residual connection input (None if no skip connection).
        Returns: X + skip_X if skip_X is provided, otherwise X.
        """
        self.setPrevIn(X)
        self.skip_X = skip_X if skip_X is not None else np.zeros_like(X)
        Y = X + self.skip_X  # Residual sum
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        """
        The gradient of the identity function is just an identity matrix.
        This means the gradient passes through unchanged for both inputs.
        Returns: Identity gradient.
        """
        batch_size, feature_dim = self.getPrevIn().shape
        return np.array([np.eye(feature_dim)] * batch_size)  # Batch of identity matrices

    def backward2(self, gradIn):
        """
        The gradient should flow **unchanged** through both the main path and the residual connection.
        gradIn: Incoming gradient from the next layer.
        Returns: Gradients for both the main path and the skip connection.
        """
        return gradIn, gradIn  # Passes gradient unchanged to both inputs
