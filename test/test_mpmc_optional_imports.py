import ast
import builtins
from pathlib import Path

import pytest


def test_mpmc_utils_remain_available_without_pyg():
    pytest.importorskip("torch")

    repository_root = Path(__file__).resolve().parent.parent
    init_path = repository_root / "qmcpy" / "__init__.py"
    init_tree = ast.parse(init_path.read_text())
    optional_import = next(
        node
        for node in init_tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(statement, ast.ImportFrom)
            and statement.module == "discrete_distribution.mpmc"
            for statement in node.body
        )
    )

    import qmcpy

    real_import = builtins.__import__

    def import_without_pyg(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "discrete_distribution.mpmc.models":
            raise ModuleNotFoundError("blocked optional dependency", name="torch_geometric")
        return real_import(name, globals, locals, fromlist, level)

    test_builtins = vars(builtins).copy()
    test_builtins["__import__"] = import_without_pyg
    namespace = {"__builtins__": test_builtins, "__package__": "qmcpy"}
    module = ast.Module(body=[optional_import], type_ignores=[])
    exec(compile(module, str(init_path), "exec"), namespace)

    assert namespace["mpmc_utils"] is qmcpy.mpmc_utils
    assert namespace["mpmc_utils"].__name__ == (
        "qmcpy.discrete_distribution.mpmc.utils"
    )
    assert "utils" not in namespace
    assert "MPMC_net" not in namespace
