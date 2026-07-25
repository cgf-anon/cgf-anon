"""Minimal desilofhe GPU wrapper required by CGF-softmax."""

import numpy as np
from desilofhe import Ciphertext, Engine


class DesiloFHEGPU:
    """Own the GPU engine and keys used by CGF-softmax."""

    def __init__(
        self,
    ) -> None:
        self.engine = Engine(use_bootstrap=True, mode="gpu")
        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relinearization_key = self.engine.create_relinearization_key(
            self.secret_key
        )
        self.rotation_key = self.engine.create_rotation_key(self.secret_key)

        self.slot_count = self.engine.slot_count
        self.start_level: int | None = None
        self.end_level: int | None = None
        self.depth: int | None = None
        self.evaluation_time: float | None = None

    def encrypt(self, value: np.ndarray, *, level: int = 10) -> Ciphertext:
        ciphertext = self.engine.encrypt(
            value, level=level, public_key=self.public_key
        )
        if self.start_level is None:
            self.start_level = ciphertext.level
        elif ciphertext.level > self.start_level:
            self.start_level = ciphertext.level
        return ciphertext

    def decrypt(self, value: Ciphertext) -> np.ndarray:
        if self.start_level is None:
            raise RuntimeError("cannot determine depth before encrypting an input")

        if self.end_level is None:
            self.end_level = value.level
        elif value.level != self.end_level:
            raise RuntimeError(
                "all output ciphertexts must have the same end level"
            )
        self.depth = self.start_level - self.end_level
        return self.engine.decrypt(value, self.secret_key)

    def reset_depth(self) -> None:
        """Clear level measurements before a new CGF-softmax evaluation."""
        self.start_level = None
        self.end_level = None
        self.depth = None
        self.evaluation_time = None

    def add(self, left, right):
        return self.engine.add(left, right)

    def subtract(self, left, right):
        return self.engine.subtract(left, right)

    def multiply(self, left, right):
        if isinstance(left, Ciphertext) and isinstance(right, Ciphertext):
            return self.engine.multiply(left, right, self.relinearization_key)
        return self.engine.multiply(left, right)

    def square(self, value):
        return self.multiply(value, value)

    def roll(self, value: Ciphertext, offset: int):
        return self.engine.rotate(value, self.rotation_key, offset)

    def constant(self, value: float) -> Ciphertext:
        message = np.full(self.slot_count, value, dtype=np.float64)
        return self.encrypt(message)
