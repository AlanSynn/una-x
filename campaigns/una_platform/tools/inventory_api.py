"""Read-only AST API census. No imports or execution of inspected source."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path


def inventory(root: Path) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    records = []
    files = []
    for path in sorted(root.rglob("*.py")):
        if any(p in {".git", "__pycache__", ".venv", "venv"} for p in path.relative_to(root).parts):
            continue
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "sha256": hashlib.sha256(raw).hexdigest()})
        module = rel[:-3].replace("/", ".")
        def walk(nodes, prefix):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_dunder = node.name.startswith("__") and node.name.endswith("__")
                    if node.name.startswith("_") and not is_dunder:
                        continue
                    records.append({"symbol": prefix + "." + node.name,
                        "kind": "callable", "signature": ast.unparse(node.args),
                        "returns": ast.unparse(node.returns) if node.returns else None,
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "source": rel, "line": node.lineno,
                        "body_requires_manual_semantic_review": True})
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    name = prefix + "." + node.name
                    records.append({"symbol": name, "kind": "class", "source": rel,
                        "line": node.lineno, "bases": [ast.unparse(b) for b in node.bases]})
                    walk(node.body, name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    records.append({"symbol": prefix, "kind": "import_or_reexport",
                        "expression": ast.unparse(node), "source": rel, "line": node.lineno})
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                        records.append({"symbol": prefix + ".__all__", "kind": "export_declaration",
                            "expression": ast.unparse(node), "source": rel, "line": node.lineno})
        walk(tree.body, module)
    if not files:
        raise ValueError("No Python source files found; empty census is not success")
    return {"schema_version": 1, "files": files, "records": records,
            "complete_semantic_census": False,
            "manual_review_required": ["dynamic exports", "inherited API", "public data attributes",
                "documented examples", "parameter effects", "unsupported stubs", "return/state/error behavior"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    print(json.dumps(inventory(args.root), indent=2))

if __name__ == "__main__":
    main()
