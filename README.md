# CGF-softmax

This anonymous repository contains the implementation of CGF-softmax. This work uses the DESILO FHE library developed by DESILO Inc. (https://desilo.ai)

## Install

```bash
pip install -e ".[test]"
```

## Run the test

```bash
pytest -q -s
```

## Example output

```text
=== CGF-softmax Test ===
Input shape:              (256, 256)
Parameters:               k=2, exp_degree=15
Multiplicative depth:     8
FHE evaluation time:      1.688084 s (after encryption -> before decryption)
Mean absolute error:      1.122200e-11
.
1 passed in 6.41s
```
