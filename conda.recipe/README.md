# Conda recipe for QMCPy

Build a local package from this repository root with:

```bash
conda build conda.recipe
```

Install the locally built package with:

```bash
conda install --use-local qmcpy
```

Render recipe metadata (without building):

```bash
conda render conda.recipe
```
