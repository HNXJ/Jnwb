# Install

## PyPI

```bash
pip install -U jnwb
```

The current library release is dataset-agnostic. The public surface is documented in [Public API](api.md) and is the contents of `jnwb.__all__`.

### Optional Acceleration Backends & Extras

`jnwb` is structured with modular extras so production workflows install only what they need:

```bash
pip install "jnwb[torch,gpu]"   # PyTorch, CuPy, and CUDA 12.x acceleration
pip install "jnwb[mcp]"         # Model Context Protocol server tooling
pip install "jnwb[docs]"        # MkDocs documentation builder
pip install "jnwb[test]"        # pytest, pytest-cov, pytest-xdist test suites
pip install "jnwb[all]"         # Complete dependency bundle
```

## Source Checkout

Clone and install an editable development environment:

```bash
git clone https://github.com/HNXJ/jnwb.git
cd jnwb
pip install -e ".[all]"
```

## Verify

Run the verification snippet in your Python environment:

```python
import jnwb

print(f"jnwb version: {jnwb.__version__}")
print(f"Public surface: {len(jnwb.__all__)} symbols")
missing = [name for name in jnwb.__all__ if not hasattr(jnwb, name)]
assert not missing, f"Unresolved public exports: {missing}"
print("Verification passed successfully.")
```
