"""AST helpers: detect bare names used but not bound (post star-import refactor)."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS_DIR = _REPO_ROOT / "src" / "spot_check" / "analysis"
_IMPORTS_PATH = _ANALYSIS_DIR / "_imports.py"


def _imports_star_names() -> frozenset[str]:
    tree = ast.parse(_IMPORTS_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    val = node.value
                    if isinstance(val, (ast.List, ast.Tuple)):
                        return frozenset(
                            elt.value
                            for elt in val.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        )
    raise RuntimeError("_imports.py: __all__ not found")


IMPORTS_STAR_NAMES = _imports_star_names()
BUILTIN_NAMES = frozenset(dir(builtins))
_ALLOW_UNBOUND = frozenset({"annotations"})


def analysis_source_paths() -> list[Path]:
    paths = sorted(_ANALYSIS_DIR.rglob("*.py"))
    return [p for p in paths if p.name not in ("__init__.py", "_core.py")]


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def undefined_bare_names(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, name) for bare Name loads not bound in scope."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert isinstance(tree, ast.Module)
    checker = _ScopeChecker(
        initial=BUILTIN_NAMES | IMPORTS_STAR_NAMES | _ALLOW_UNBOUND | _module_level_names(tree),
        star_from_imports=IMPORTS_STAR_NAMES,
    )
    checker.visit(tree)
    return checker.undefined


class _ScopeChecker(ast.NodeVisitor):
    def __init__(
        self,
        *,
        initial: set[str],
        star_from_imports: frozenset[str],
    ) -> None:
        self._star = star_from_imports
        self._scopes: list[set[str]] = [set(initial)]
        self.undefined: list[tuple[int, str]] = []

    @property
    def _scope(self) -> set[str]:
        return self._scopes[-1]

    def _bind(self, name: str) -> None:
        if name:
            self._scope.add(name)

    def _bind_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_target(elt)

    def _use(self, name: str, lineno: int) -> None:
        if name not in self._scope:
            self.undefined.append((lineno, name))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.endswith("._imports"):
            for alias in node.names:
                if alias.name == "*":
                    self._scope.update(self._star)
                    continue
                self._bind(alias.asname or alias.name)
        else:
            for alias in node.names:
                if alias.name == "*":
                    continue
                self._bind(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._use(node.id, node.lineno or 0)
        elif isinstance(node.ctx, ast.Store):
            self._bind(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name)
        self._scopes.append(set(self._scope))
        args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        if node.args.vararg:
            args.append(node.args.vararg)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        for arg in args:
            self._bind(arg.arg)
        self.generic_visit(node)
        self._scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)
        self._scopes.append(set(self._scope))
        self.generic_visit(node)
        self._scopes.pop()

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._bind(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._bind(name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._bind(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._bind_target(node.target)
        self.visit(node.iter)
        for clause in node.ifs:
            self.visit(clause)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        for gen in node.generators:
            self.visit(gen)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        for gen in node.generators:
            self.visit(gen)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        for gen in node.generators:
            self.visit(gen)
        self.visit(node.key)
        self.visit(node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        for gen in node.generators:
            self.visit(gen)
        self.visit(node.elt)
