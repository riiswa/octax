# Installation

## From PyPI

```bash
pip install octax
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install octax
```

## GPU acceleration (recommended)

For GPU acceleration, install JAX with CUDA support after installing Octax:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Adjust the CUDA version (e.g. `cuda11_pip`, `cuda12_pip`) to match your system.

## From source

```bash
git clone https://github.com/riiswa/octax.git
cd octax
pip install -e .
```

With uv:

```bash
uv sync
```

## Optional dependencies

- **GUI** (rendering, visualization): `pip install octax[gui]`
- **Development** (tests, coverage): `pip install octax[dev]`
- **Documentation** (build docs locally): `pip install octax[docs]`

Install everything:

```bash
pip install octax[all]
```

## Requirements

- Python >= 3.10
- JAX >= 0.4.20
- NumPy >= 1.24.0
- Flax >= 0.8.0
