# AgBLR
Comparing Bayesian linear regression with industry standard linear mixed model framework for agricultural research crop yield trial analyses

## Dev Stuff

```shell
git clone https://github.com/jeffersonfparil/AgBLR.git
cd AgBLR
uv init .
uv add --dev ruff pytest
uv add numpy scipy matplotlib
uv add nptyping
# uv sync
mkdir scripts
mkdir scripts/py
mkdir scripts/jl
mkdir scripts/R
touch scripts/__init__.py
touch scripts/py/__init__.py
touch scripts/jl/__init__.py
touch scripts/R/__init__.py
mkdir tests
touch tests/__init__.py
touch tests/tests.py
# Development loop
source .venv/bin/activate
python -i scripts/py/simulate.py
uv run ruff format
uv run ruff check --fix
uv run pytest -v
```