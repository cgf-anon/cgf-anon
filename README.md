# CGF-softmax

This anonymous repository contains only the **GPU implementation** of CGF-softmax used during the paper review. This work uses the GPU version of the DESILO FHE library developed by DESILO Inc. (https://desilo.ai)

## Install

The package requires an NVIDIA GPU and a CUDA 12.4-compatible environment.

```bash
pip install -e ".[test]"
```

## Run the test

```bash
pytest -q -s
```

## Example output

```text
=== CGF-softmax GPU Test ===
Input shape:              (256, 256)
Parameters:               k=2, exp_degree=15
Multiplicative depth:     8
FHE evaluation time:      2.159405 s (after encryption -> before decryption)
Mean absolute error:      1.122132e-11
.
1 passed in 5.26s
```
