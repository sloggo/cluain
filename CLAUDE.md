# CLAUDE.md

## Project Overview

Cluain is a Semantic Source Code Duplication (SSCD) detector for C/C++ codebases. It uses CodeBERT embeddings to detect semantically similar code clones, classifying them into Types 1-4.

## Quick Start

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

```
cluain/
├── __init__.py      # Package exports: Analyser, PRAnalyser
├── parser.py        # Tree-sitter C/C++ function extraction
├── embeddings.py    # CodeBERT embedding generation
├── similarity.py    # Cosine similarity & clone classification (T1-T4)
├── analyser.py      # Main codebase analysis orchestration
├── pr_analyser.py   # PR/CI-specific analysis
├── git_utils.py     # Git operations (diff, worktree, clone)
├── formatters.py    # Output formatting (JSON, GitHub, CI)
└── cli.py           # Command-line interface
```

## Module Responsibilities

### parser.py
- `CodeParser`: Extracts function definitions from C/C++ files using tree-sitter
- `scan_directory()`: Recursively scans codebase for functions

### embeddings.py
- `CodeEmbedder`: Generates embeddings using CodeBERT model
- Configurable device (cpu/mps), threads, batch size

### similarity.py
- `compute_similarity_matrix()`: Cosine similarity between embeddings
- `classify_clone_type()`: Determines T1/T2/T3/T4 based on:
  - T1: Exact match after whitespace normalization
  - T2: Same structure with renamed identifiers
  - T3: High token similarity (>0.7)
  - T4: Semantic similarity only
- `find_duplicates()`: Batch duplicate detection

### analyser.py
- `Analyser`: Main class for full codebase analysis
- Orchestrates parser → embedder → similarity

### pr_analyser.py
- `PRAnalyser`: Analyses PRs for introduced duplication
- Compares changed functions against entire codebase
- Tracks duplicates within PR vs against existing code

### formatters.py
- `format_json()`: JSON output
- `format_github_comment()`: Markdown for PR comments
- `format_ci_output()`: Console-friendly output

### git_utils.py
- `get_changed_files()`: Files changed between commits
- `create_worktree()`: Create worktree for historical analysis
- `clone_repo()`: Clone repositories

## Usage

### Codebase Analysis
```python
from cluain import Analyser

analyser = Analyser(threshold=0.95, device='cpu')
result = analyser.analyse("./path/to/repo", excluded_paths=["/tests"])
```

### PR Analysis (CLI)
```bash
python -m cluain.cli --repo . --base main --head feature --output github
```

### GitHub Action
```yaml
- uses: sloggo/cluain@main
  with:
    threshold: '0.95'
    comment-on-pr: 'true'
```

## Clone Types

| Type | Description | Detection |
|------|-------------|-----------|
| T1 | Exact copy | Normalized string match |
| T2 | Renamed identifiers | Token sequence match |
| T3 | Modified statements | Token similarity >0.7 |
| T4 | Semantic only | Embedding similarity only |
