"""CGF-softmax implemented with desilofhe on GPU."""

from .core import cgf_softmax
from .gpu import DesiloFHEGPU

__all__ = ["DesiloFHEGPU", "cgf_softmax"]
