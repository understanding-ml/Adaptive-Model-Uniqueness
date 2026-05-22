# isqed/real_world.py
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from isqed.core import ModelUnit, Intervention
from typing import Tuple, Optional
import torch.nn as nn

class HuggingFaceWrapper(ModelUnit):
    """
    Wraps a Hugging Face model for ISQED auditing.
    Task: Sentiment Analysis (Binary Classification).
    Output: Probability of 'Positive' class.
    """
    def __init__(self, model_name, device='cpu'):
        super().__init__(name=model_name)
        self.device = device
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval() # Set to inference mode

    def _forward(self, text_input):
        # text_input is a string (already perturbed)
        inputs = self.tokenizer(
            text_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # We assume binary classification (SST-2). 
        # Return the probability of label 1 (Positive) as the scalar response Y.
        # Shape: (1,) scalar
        return probs[0, 1].item()

    
class MaskingIntervention(Intervention):
    def __init__(self, mask_token: str = "[MASK]"):
        self.mask_token = mask_token

    def apply(self, text: str, theta: float, seed: int = None) -> str:
        """
        Apply masking with a fixed seed to ensure that target and peers
        see exactly the same corrupted input for a given (text, theta).

        Args:
            text: original input sentence
            theta: masking ratio in [0, 1]
            seed: integer seed to make the masking pattern deterministic

        Returns:
            perturbed_text: sentence with a fraction of tokens replaced by [MASK]
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        words = text.split()
        n = len(words)
        n_mask = int(n * theta)

        if n_mask > 0 and n > 0:
            mask_idx = rng.choice(n, n_mask, replace=False)
            for i in mask_idx:
                words[i] = self.mask_token
        return " ".join(words)


class ImageModelWrapper:
    """
    Lightweight wrapper around a torchvision image classifier.

    The wrapper takes as input a tuple (x, y):
      - x: tensor of shape (3, H, W), already normalized
      - y: integer class label (ImageNet index)

    It outputs a scalar g(x) depending on `mode`:

      - "margin":        logit_y - max_{k != y} logit_k           (原始定义)
      - "tanh_margin":   tanh( (logit_y - max_{others}) / tau )   (压缩极端值)
      - "p_true":        softmax(logits)_y                        (true-class prob)
      - "signed_pred":   +p_hat if correct, -p_hat if incorrect   (自信对/错)
    """

    def __init__(
        self,
        model: nn.Module,
        name: str,
        device: str,
        mode: str = "tanh_margin",
        tau: float = 10.0,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.name = name
        self.device = device
        self.mode = mode
        self.tau = tau

    @torch.no_grad()
    def _forward(self, sample: Tuple[torch.Tensor, int]) -> float:
        x, y = sample
        x = x.to(self.device)
        y_int = int(y)

        logits = self.model(x.unsqueeze(0))  # (1, C)
        logits = logits[0]  # (C,)
        if logits.ndim != 1:
            logits = logits.view(-1)

        if self.mode == "margin":
            logit_y = logits[y_int].item()
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[y_int] = False
            max_other = logits[mask].max().item()
            return float(logit_y - max_other)

        # softmax once,用于下面几种模式
        probs = torch.softmax(logits, dim=0)
        p_true = probs[y_int].item()
        pred_idx = int(torch.argmax(probs).item())
        p_pred = probs[pred_idx].item()

        if self.mode == "p_true":
            return float(p_true)

        if self.mode == "tanh_margin":
            logit_y = logits[y_int]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[y_int] = False
            max_other = logits[mask].max()
            margin = (logit_y - max_other) / self.tau
            return float(torch.tanh(margin).item())

        if self.mode == "signed_pred":
            if pred_idx == y_int:
                return float(p_pred)   # confident correct: positive
            else:
                return float(-p_pred)  # confident wrong: negative

        # fallback: original margin
        logit_y = logits[y_int].item()
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[y_int] = False
        max_other = logits[mask].max().item()
        return float(logit_y - max_other)


class AdversarialFGSMIntervention:
    """
    Simple FGSM adversarial intervention driven by a single reference model.

    Input sample is (x, y):
      - x: normalized image tensor (3, H, W)
      - y: integer class label

    We compute:
        x_adv = x + epsilon * sign(grad_x L(ref_model(x), y))

    and return (x_adv, y), so that downstream wrappers can still see
    the true label y when computing p_correct.
    """

    def __init__(
        self,
        ref_model: nn.Module,
        device: str,
        loss_fn: Optional[nn.Module] = None,
    ):
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        self.device = device
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def apply(
        self,
        sample: Tuple[torch.Tensor, int],
        epsilon: float,
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply a single-step FGSM attack with magnitude epsilon.
        """
        x, y = sample
        x = x.to(self.device)
        y_tensor = torch.tensor([y], device=self.device, dtype=torch.long)

        x_adv = x.clone().detach().unsqueeze(0)
        x_adv.requires_grad = True

        logits = self.ref_model(x_adv)
        loss = self.loss_fn(logits, y_tensor)
        self.ref_model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv + epsilon * grad_sign

        # Clamp to a reasonable range in normalized space
        x_adv = torch.clamp(x_adv, -3.0, 3.0)

        x_adv = x_adv.detach().squeeze(0).cpu()
        return (x_adv, y)
    

class ImageIdentityIntervention:
    """
    Identity intervention for images.

    It ignores theta and seed and simply returns the original (x, y) pair.
    This lets us reuse the DISCO / Ecosystem machinery with theta = 0.
    """

    def apply(self, sample: Tuple[torch.Tensor, int], theta: float, seed: int = 0):
        return sample
    

class TabularModelWrapper:
    """
    Lightweight wrapper around a scikit-learn regression model.

    Input to `_forward` is a 1D numpy array x of shape (D,).
    The output is a scalar prediction y_hat in R.
    """

    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def _forward(self, x: np.ndarray) -> float:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=float)
        x_2d = x.reshape(1, -1)
        y_hat = self.model.predict(x_2d)[0]
        return float(y_hat)


class TabularIdentityIntervention:
    """
    Identity intervention for tabular data.

    It ignores theta and seed and simply returns the original feature vector.
    """

    def apply(self, x: np.ndarray, theta: float, seed: int = 0) -> np.ndarray:
        return x