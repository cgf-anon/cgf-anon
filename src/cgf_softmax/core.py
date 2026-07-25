"""CGF-softmax homomorphic computation."""

import time

import numpy as np
from numpy.polynomial.chebyshev import cheb2poly, chebfit

from .packing import gap_pack, gap_unpack
from .polynomial import evaluate_polynomial


def exp_chebyshev_coeff(a=-8, b=0, degree=15, num_samples=1000):
    x = np.linspace(a, b, num_samples)
    x_scaled = (2 * x - (a + b)) / (b - a)
    return cheb2poly(chebfit(x_scaled, np.exp(x), degree))


def cgf_softmax(
    engine,
    input_data: np.ndarray,
    *,
    k: int = 2,
    exp_degree: int = 15,
) -> np.ndarray:
    """Evaluate CGF-softmax on a two-dimensional array of row vectors."""
    if not isinstance(input_data, np.ndarray) or input_data.ndim != 2:
        raise ValueError("input_data must be a two-dimensional NumPy array")
    if input_data.size == 0:
        raise ValueError("input_data must not be empty")
    if exp_degree not in (7, 15, 31, 63, 127):
        raise ValueError("exp_degree must be one of 7, 15, 31, 63, or 127")

    # Exp
    a=-7
    b=1
    exp_coeff = exp_chebyshev_coeff(a=a,b=b,degree=exp_degree)
    
    # Engine reset 
    engine.reset_depth()
    
    # Scaled input
    N1 = input_data.shape[0]
    n = input_data.shape[1]
    scaled_input = input_data / n
    
    # Plaintext Setting
    scaled_x, g = gap_pack(scaled_input, slot_count=2**15)
    m = len(scaled_x)
    s = 2**15
    lnn_over_2k = 2.0 * np.log(n) / (2 ** k) / (b-a)
    
    for i in range(m):
        scaled_x[i] = engine.encrypt(scaled_x[i], level = 10)

    # FHE CGF-softmax start
    evaluation_start = time.perf_counter()
    
    def fhe_sum(cts):
        if len(cts) == 1:
            return cts[0]
        result = engine.add(cts[0],cts[1])
        for i in range(2, m):
            result = engine.add(result, cts[i])
        return result

    def fhe_rot_sum(ct):
        rot_idx = int(g)
        while rot_idx < s:
            rotated = engine.roll(ct, rot_idx)
            ct = engine.add(ct, rotated)
            rot_idx *= 2
        return ct
    
    summed = fhe_sum(scaled_x)
    mu = fhe_rot_sum(summed)
    
    x = [None]*m
    for i in range(m):
        x[i] = engine.multiply(scaled_x[i], int(n))
        x[i] = engine.subtract(x[i], mu)

    squared = [None]*m
    for i in range(m):
        squared[i] = engine.square(x[i])
    
    n_var = fhe_rot_sum(fhe_sum(squared))
    
    # Scale down
    for i in range(len(x)):
        x[i] = engine.multiply(x[i], 2.0 /(2**k)/(b-a))
    scaled_var = engine.multiply(n_var, 2.0/ (2*n*(2**k))/(b-a))

    for i in range(len(x)):
        x[i] = engine.subtract(x[i], scaled_var)
        x[i] = engine.subtract(x[i], lnn_over_2k)

    # Exponential computation
    polynomial_input = [engine.add(part,-(a + b) / (b - a)) for part in x]
    result = evaluate_polynomial(engine, polynomial_input, exp_coeff)

    # Scale up
    for i in range(k):
        for j in range(len(result)):
            result[j] = engine.square(result[j])

    engine.evaluation_time = time.perf_counter() - evaluation_start

    decrypted = [engine.decrypt(part) for part in result]
    return gap_unpack(decrypted, (N1, n), engine.slot_count)
