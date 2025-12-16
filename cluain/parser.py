"""
Code parser using tree-sitter for C/C++ function extraction.
"""

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
from pathlib import Path


class CodeParser:
    """Extracts function definitions from C/C++ source files."""

    EXTENSIONS = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx'}
    CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.hpp', '.hxx', '.h'}

    def __init__(self, minimum_block_size: int = 80):
        self.minimum_block_size = minimum_block_size
        self.c_language = Language(tsc.language())
        self.cpp_language = Language(tscpp.language())
        self.c_parser = Parser(self.c_language)
        self.cpp_parser = Parser(self.cpp_language)

    def _get_parser(self, file_path: Path) -> Parser:
        """Select appropriate parser based on file extension."""
        if file_path.suffix in self.CPP_EXTENSIONS:
            return self.cpp_parser
        return self.c_parser

    def extract_functions(self, file_path: Path) -> list:
        """
        Extract function definitions from a single file.

        Returns:
            List of dicts with keys: code, type, start_line, end_line, file, size
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            parser = self._get_parser(file_path)
            tree = parser.parse(content)

            blocks = []
            self._traverse(tree.root_node, file_path, blocks)
            return blocks

        except Exception:
            return []

    def _traverse(self, node, file_path: Path, blocks: list):
        """Recursively traverse AST to find function definitions."""
        if node.type == 'function_definition':
            code = node.text.decode('utf-8', errors='ignore')
            if len(code.strip()) > 50:
                blocks.append({
                    'code': code,
                    'type': 'function',
                    'start_line': node.start_point[0] + 1,
                    'end_line': node.end_point[0] + 1,
                    'file': str(file_path),
                    'size': len(code)
                })

        for child in node.children:
            self._traverse(child, file_path, blocks)

    def scan_directory(self, root_dir: str, excluded_paths: list = None) -> list:
        """
        Scan directory for C/C++ files and extract all functions.

        Args:
            root_dir: Root directory to scan
            excluded_paths: List of paths to exclude (e.g., ["/tests", "/bench"])

        Returns:
            List of all code blocks found
        """
        root = Path(root_dir).resolve()
        excluded = self._resolve_excluded_paths(root, excluded_paths or [])

        all_blocks = []
        for ext in ['*.c', '*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp', '*.hxx']:
            for file_path in root.rglob(ext):
                file_path = file_path.resolve()

                if self._should_skip(file_path, excluded):
                    continue

                blocks = self.extract_functions(file_path)
                all_blocks.extend(blocks)

        return all_blocks

    def _resolve_excluded_paths(self, root: Path, excluded_paths: list) -> list:
        """Convert excluded path strings to resolved Path objects."""
        return [(root / ex.lstrip("/")).resolve() for ex in excluded_paths]

    def _should_skip(self, file_path: Path, excluded: list) -> bool:
        """Check if file should be skipped based on exclusion rules."""
        for ex in excluded:
            if ex.is_dir() and file_path.is_relative_to(ex):
                return True
            if file_path == ex:
                return True
        return False
