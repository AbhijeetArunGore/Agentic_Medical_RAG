# Contributing Guidelines

Thank you for your interest in improving **Agentic Medical RAG**! Please review the development conventions and standards below.

---

## 1. Local Environment Setup

### Windows PowerShell
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install production and development dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Configure environment
Copy-Item .env.example .env
```

### macOS / Linux
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Configure environment
cp .env.example .env
```

---

## 2. Code Quality & Formatting

Before opening a pull request, run the formatters and linters:

```bash
# Code formatting with Black
black app tests main.py evaluation benchmarks

# Import sorting with isort
isort app tests main.py evaluation benchmarks

# Flake8 linting
flake8 app tests main.py evaluation benchmarks --max-line-length=120 --extend-ignore=E402,W293,E501,E722,F401,F841
```

---

## 3. Running Tests & Benchmarks

```bash
# Run pytest test suite (54 tests)
python -m pytest tests/ -v

# Run comparative retrieval benchmark
python -m evaluation.runners.evaluate_retrieval

# Run answer quality evaluation
python -m evaluation.runners.evaluate_answers

# Run API latency benchmark
python benchmarks/run_benchmark.py --iterations 50
```

---

## 4. Git & Branching Hygiene

- **Branch naming**: `feat/description`, `fix/description`, `chore/description`
- **Secrets Policy**: NEVER commit real API keys, credentials, or private patient health data.
- **Commit Messages**: Follow Conventional Commits (e.g. `feat(retriever): add BM25 lexical fusion`).
