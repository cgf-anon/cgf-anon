# CGF-softmax

This anonymous repository contains only the **GPU implementation** of CGF-softmax used during the paper review. This work uses the DESILO FHE library developed by DESILO Inc. (https://desilo.ai)

## Install

The package requires an NVIDIA GPU and a CUDA 12.4-compatible environment.

```bash
pip install -e ".[test]"
```

## Run the test

```bash
pytest -q -s
```
