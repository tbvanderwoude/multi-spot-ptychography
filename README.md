# master

This project uses [**uv**](https://github.com/astral-sh/uv) for Python environment and dependency management.

## Quick start

```bash
# 1. Install uv (if not yet installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone this repository
git clone https://github.com/tbvanderwoude/master.git
cd huygens

# 3. Create environment and install dependencies with cuda12 (cuda13 is also possible)
uv sync --extra cuda12

# 4. Activate created environment on Linux
source .venv/bin/activate
```

## Installation

To install `master` as editable package for development, simply run

```
pip install -e .
```

## Pre-commit hooks

To activate precommit hooks, use

```bash
pre-commit install
```
