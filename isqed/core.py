from abc import ABC, abstractmethod
import numpy as np

class Intervention(ABC):
    """Abstract intervention T(theta, x)"""
    @abstractmethod
    def apply(self, x, theta: float):
        pass

class Scalarizer(ABC):
    """Abstract scalarised function g(y) -> scalar"""
    @abstractmethod
    def __call__(self, raw_output) -> float:
        pass

class ModelUnit(ABC):
    """Unified model unit for ISQED framework"""
    def __init__(self, name: str, scalarizer: Scalarizer = None):
        self.name = name
        self.scalarizer = scalarizer

    @abstractmethod
    def _forward(self, input_data):
        pass

    def query(self, x, theta, intervention: Intervention) -> float:
        """ISQED Core Operation: In-Silico Query"""
        perturbed_x = intervention.apply(x, theta)
        raw_out = self._forward(perturbed_x)
        return self.scalarizer(raw_out) if self.scalarizer else raw_out