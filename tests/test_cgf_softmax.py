import numpy as np

from cgf_softmax import DesiloFHECPU, cgf_softmax



def test_cgf_softmax_on_cpu():
    k = 2
    exp_degree = 15
    cpu_engine = DesiloFHECPU()
    rng = np.random.default_rng(42)
    values = rng.normal(0.0, 1.0, size=(256, 256)) 

    actual = cgf_softmax(
        cpu_engine,
        values,
        k=k,
        exp_degree=exp_degree,
    )

    mean = np.mean(values, axis=1, keepdims=True)
    variance = np.var(values, axis=1, keepdims=True)
    expected = np.exp(values - mean - 0.5 * variance) / values.shape[1]

    mean_error = np.mean(np.abs(expected - actual))

    print("\n=== CGF-softmax Test ===")
    print(f"Input shape:              {values.shape}")
    print(f"Parameters:               k={k}, exp_degree={exp_degree}")
    print(f"Multiplicative depth:     {cpu_engine.depth}")
    print(
        "FHE evaluation time:      "
        f"{cpu_engine.evaluation_time:.6f} s "
        "(after encryption -> before decryption)"
    )
    print(f"Mean absolute error:      {mean_error:.6e}")

    assert mean_error < 1e-9
