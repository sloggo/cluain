"""
Main entry point for running cluain as a module.

Usage:
    python -m cluain pr --repo . --base main --head feature
    python -m cluain analyse ./path/to/repo
    python -m cluain history github.com/user/repo --years 3
    python -m cluain batch repositories.csv --output-dir ./results
"""

import sys


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1]

    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == 'pr':
        from .cli import main_pr
        main_pr()
    elif command == 'analyse':
        from .cli import main_analyse
        main_analyse()
    elif command == 'history':
        from .cli import main_historical
        main_historical()
    elif command == 'batch':
        from .cli import main_batch
        main_batch()
    elif command in ['-h', '--help']:
        print_help()
    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)


def print_help():
    print("""Cluain - Semantic Code Clone Detector for C/C++

Usage: python -m cluain <command> [options]

Commands:
    pr        Analyse PR for code duplication (CI/CD)
    analyse   Analyse a codebase for duplication
    history   Track duplication history over git commits
    batch     Batch analyse multiple repos from CSV

Examples:
    python -m cluain pr --repo . --base main --head feature --output github
    python -m cluain analyse ./myproject --output results.json
    python -m cluain history github.com/user/repo --years 3
    python -m cluain batch repositories.csv --output-dir ./results

Run 'python -m cluain <command> --help' for command-specific options.
""")


if __name__ == '__main__':
    main()
