# Installation

## Install from [PyPI](https://pypi.org/project/gaussquad/)

```bash
pip install gaussquad
```

## Install from GitHub

```bash
pip install git+https://github.com/thatmariia/gaussquad.git
```
> **Requirements:** Python 3.13 or newer, numpy, scipy

---

## Development & docs 

If you plan to contribute, test, or build docs:

0. Create a virtual environment 
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
1. Clone the repository:
```bash
git clone https://github.com/thatmariia/gaussquad.git
cd gaussquad
```
2. Install in editable mode with development and docs dependencies
```bash
pip install -e ".[dev,docs]"
```

### Run tests
```bash
pytest
```

### Code style checks
```bash
black --check src/
flake8 src/
```

### Build docs
```bash
cd docs
make html
```

### View docs
```bash
open docs/build/html/index.html  # On macOS
# or:
python -m http.server --directory docs/build/html
```
Then open your browser and go to `http://localhost:8000/`.