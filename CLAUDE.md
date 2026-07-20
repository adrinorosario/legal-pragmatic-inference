# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository appears to be a legal inference project involving dataset construction and analysis. It uses DVC (Data Version Control) for managing datasets and contains Jupyter notebooks for analysis.

## Key Directories

- `datasets/` - Contains dataset files
- `dataset construction notebooks/` - Jupyter notebooks for dataset creation and analysis
- `docs/` - Documentation
- `.dvc/` - DVC configuration and tracked files
- `src/models/` - Model schemas (if present)

## Development Commands

### DVC Operations
```bash
# Pull latest datasets
dvc pull

# Push datasets to remote storage
dvc push

# Reproduce pipeline
dvc repro
```

### Python/Jupyter Development
```bash
# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt

# Run a Jupyter notebook
jupyter nbconvert --to notebook --execute notebook.ipynb

# Run Python tests (if tests exist)
pytest
```

## Architecture Notes

- Uses DVC for data versioning and pipeline management
- Analysis workflows are implemented as Jupyter notebooks
- Git is used for source control
- Datasets are stored via DVC and may include large files

## Common Tasks

1. **Adding new datasets**: Use `dvc add <file>` then commit both the file and .dvc metadata
2. **Running analysis**: Open notebooks in `dataset construction notebooks/` directory
3. **Pipeline reproduction**: Run `dvc repro` to regenerate outputs from inputs