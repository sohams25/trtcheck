# Install

## From PyPI

```bash
pip install trtcheck
```

Requires Python 3.10+.

Verify:

```bash
trtcheck --version
```

## From source

```bash
git clone https://github.com/sohams25/trtcheck.git
cd trtcheck
pip install -e ".[dev]"
```

The `[dev]` extras include `pytest`, `mypy`, `black`, and `isort`.

## Development environment notes

If your shell has ROS or another distro that injects into `PYTHONPATH`,
strip it before running tests:

```bash
./scripts/run-tests.sh
```

That wrapper exists for exactly this reason.
