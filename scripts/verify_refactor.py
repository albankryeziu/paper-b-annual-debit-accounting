"""Check that the readable source retains the recorded Python syntax trees.

Module documentation and explicit expansion of engineering_common imports are
excluded. Trimmed modules compare the listed retained definitions and constants,
with their imports checked separately. Function bodies, comparison signs,
solver settings and arithmetic order remain checked. This is not a new numerical
qualification or an equivalence proof for changed filesystem paths.
"""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_tree(text, engineering=False, retained_names=None):
    """Normalize the two documented editorial changes only."""

    class EditorialChanges(ast.NodeTransformer):
        def visit_Module(self, node):
            node = self.generic_visit(node)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
            return node

        def visit_ImportFrom(self, node):
            if engineering and node.module == "engineering_common":
                return None
            return node

    tree = EditorialChanges().visit(ast.parse(text))
    if retained_names is not None:
        selected = set(retained_names)
        tree.body = [node for node in tree.body if node_names(node) & selected]
    # Python 3.12 adds empty type_params fields to older source syntax.
    for node in ast.walk(tree):
        if hasattr(node, "type_params"):
            if node.type_params:
                raise ValueError("Unexpected Python 3.12 type parameter syntax")
            node._fields = tuple(field for field in node._fields if field != "type_params")
    return ast.dump(tree, include_attributes=False)


def node_names(node):
    """Names defined by a top-level definition or simple constant assignment."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)}
    return set()


def verify():
    """Verify the mapped sources and the explicitly expanded import bindings."""
    manifest = json.loads((ROOT / "reproducibility/SOURCE_MAP.json").read_text())
    for row in manifest["files"]:
        text = (ROOT / row["path"]).read_text(encoding="utf-8-sig")
        engineering = row["origin"] == "engineering"
        retained = row.get("retained_names")
        if retained is not None:
            parsed = ast.parse(text)
            for index, node in enumerate(parsed.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                if (
                    index == 0
                    and isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    continue
                if not node_names(node) or not node_names(node) <= set(retained):
                    raise ValueError("Unexpected retained-module statement: " + row["path"])
            imports = ast.Module(
                body=[n for n in parsed.body if isinstance(n, (ast.Import, ast.ImportFrom))],
                type_ignores=[],
            )
            if hashlib.sha256(ast.dump(imports).encode()).hexdigest() != row["imports_sha256"]:
                raise ValueError("Retained-module imports changed: " + row["path"])
        tree = source_tree(text, engineering, retained)
        actual = hashlib.sha256(tree.encode()).hexdigest()
        if actual != row["ast_sha256"]:
            raise ValueError("Source structure changed: " + row["path"])
        if engineering:
            if any(
                alias.asname is not None
                for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.ImportFrom) and node.module == "engineering_common"
                for alias in node.names
            ):
                raise ValueError("Unexpected common import alias: " + row["path"])
            bindings = [
                alias.name
                for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.ImportFrom) and node.module == "engineering_common"
                for alias in node.names
            ]
            if bindings != row["explicit_common_bindings"]:
                raise ValueError("Explicit common bindings changed: " + row["path"])
    return len(manifest["files"])


if __name__ == "__main__":
    print("PASS:", verify(), "mapped source structures retained.")
