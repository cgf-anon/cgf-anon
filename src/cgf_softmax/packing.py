"""Gap packing used by CGF-softmax."""

import numpy as np


def gap_pack(values: np.ndarray, slot_count: int) -> tuple[list[np.ndarray], int]:
    if not isinstance(values, np.ndarray) or values.ndim != 2:
        raise ValueError("values must be a two-dimensional NumPy array")

    row_count, column_count = values.shape
    ciphertext_count = int(np.ceil(values.size / slot_count))
    gap = ciphertext_count * slot_count // column_count

    packed = [np.zeros(slot_count, dtype=np.float64) for _ in range(ciphertext_count)]
    for row in range(row_count):
        for column in range(column_count):
            packed[column * gap // slot_count][row + (column * gap) % slot_count] = (
                values[row, column]
            )
    return packed, gap


def gap_unpack(
    packed: list[np.ndarray],
    shape: tuple[int, int],
    slot_count: int,
) -> np.ndarray:
    row_count, column_count = shape
    gap = len(packed) * slot_count // column_count
    values = np.zeros(shape, dtype=np.float64)

    for row in range(row_count):
        for column in range(column_count):
            values[row, column] = packed[column * gap // slot_count][
                row + (column * gap) % slot_count
            ]
    return values
