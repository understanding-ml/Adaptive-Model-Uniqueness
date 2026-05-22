from isqed.core import ModelUnit, Intervention
import numpy as np

class LinearStructuralModel(ModelUnit):
    """
    Implements Y = beta^T * x + noise
    Corresponds to the Local Linear Structural Model in the paper.
    """
    def __init__(self, dim, beta=None, noise_std=0.0):
        super().__init__(name="LinearSynthetic")
        self.dim = dim
        if beta is None:
            self.beta = np.random.randn(dim)
            # Normalize to unit sphere for stability
            self.beta /= np.linalg.norm(self.beta)
        else:
            self.beta = np.array(beta)
        self.noise_std = noise_std

    def _forward(self, input_vec):
        # input_vec shape: (batch, dim)
        clean_y = input_vec @ self.beta
        noise = np.random.randn(*clean_y.shape) * self.noise_std
        return clean_y + noise

class NoiseIntervention(Intervention):
    """For example, scaling certain dimensions of x by theta"""
    def apply(self, x, theta):
        # For example, scale certain dimensions of x by theta
        return x * theta