import json

from scripts.flatten_qmcpy_imports import flatten_imports, main


def _nested_import(module, imported):
    return f"from {'qmcpy.' + module} import {imported}"


def test_flatten_imports_preserves_imported_names_and_formatting():
    source = (
        _nested_import("integrand", "Keister")
        + "\n"
        + _nested_import("discrete_distribution.lattice", "Lattice as LD")
        + "\nfrom qmcpy import DigitalNetB2\nimport qmcpy.util\n"
    ).encode()

    updated, count = flatten_imports(source)

    assert count == 3
    assert updated == (
        b"from qmcpy import DigitalNetB2, Keister, Lattice as LD\n"
        b"import qmcpy.util\n"
    )


def test_flatten_imports_preserves_private_modules_and_names():
    source = (
        _nested_import("_internal._helpers", "PublicHelper")
        + "\n"
        + _nested_import(
            "true_measure.uniform_triangle",
            "UniformTriangle, _UniformTriangleAdapter",
        )
        + "\n"
        + _nested_import(
            "true_measure.copula",
            "(\n    AbstractCopula,\n    _validate_dimension,\n)",
        )
        + "\n"
        + _nested_import("integrand", "Keister")
        + "\n"
    ).encode()

    updated, count = flatten_imports(source)

    assert count == 1
    assert updated == source.replace(
        _nested_import("integrand", "Keister").encode(),
        b"from qmcpy import Keister",
    )


def test_private_module_import_separates_public_import_groups():
    source = (
        b"from qmcpy import Zeta\n"
        b"from qmcpy._internal._helpers import PublicHelper\n"
        b"from qmcpy import Alpha\n"
    )

    updated, count = flatten_imports(source)

    assert count == 0
    assert updated == source


def test_flatten_imports_deduplicates_same_scope_star_imports():
    source = (
        b"from qmcpy import *\n"
        + (_nested_import("util", "*") + "\n").encode()
        + b"\n"
        + b"    from qmcpy import *\n"
    )

    updated, count = flatten_imports(source)

    assert count == 2
    assert updated == (
        b"from qmcpy import *\n"
        b"\n"
        b"    from qmcpy import *\n"
    )


def test_flatten_imports_deduplicates_notebook_star_imports():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    _nested_import("integrand", "*") + "\n",
                    _nested_import("true_measure", "*"),
                ],
            }
        ]
    }
    source = json.dumps(notebook, indent=1).encode()

    updated, count = flatten_imports(source)

    assert count == 3
    assert json.loads(updated)["cells"][0]["source"] == ["from qmcpy import *"]


def test_flatten_imports_combines_and_alphabetizes_named_imports():
    source = (
        b"from qmcpy import Zeta,Beta\n"
        b"from qmcpy import Alpha\n"
        b"\n"
        b"from qmcpy import Gamma\n"
    )

    updated, count = flatten_imports(source)

    assert count == 1
    assert updated == (
        b"from qmcpy import Alpha, Beta, Zeta\n"
        b"\n"
        b"from qmcpy import Gamma\n"
    )
    assert flatten_imports(updated) == (updated, 0)


def test_flatten_imports_combines_parenthesized_and_single_line_imports():
    source = b"""from qmcpy import (
    KernelDigShiftInvar,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDigShiftInvarCombined,
    KernelShiftInvar,
    KernelShiftInvarCombined,
)
from qmcpy import tf_exp_eps, tf_exp_eps_inv
"""

    updated, count = flatten_imports(source)

    assert count == 1
    assert updated == b"""from qmcpy import (
    KernelDigShiftInvar,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDigShiftInvarCombined,
    KernelShiftInvar,
    KernelShiftInvarCombined,
    tf_exp_eps,
    tf_exp_eps_inv,
)
"""
    assert flatten_imports(updated) == (updated, 0)


def test_flatten_imports_combines_only_within_the_same_scope():
    source = (
        b"if enabled:\n"
        b"    from qmcpy import Zeta\n"
        b"    from qmcpy import Alpha as First\n"
        b"else:\n"
        b"    from qmcpy import Beta\n"
        b"from qmcpy import _Private\n"
        b"from qmcpy import Gamma  # keep this comment\n"
    )

    updated, count = flatten_imports(source)

    assert count == 1
    assert updated == (
        b"if enabled:\n"
        b"    from qmcpy import Alpha as First, Zeta\n"
        b"else:\n"
        b"    from qmcpy import Beta\n"
        b"from qmcpy import _Private\n"
        b"from qmcpy import Gamma  # keep this comment\n"
    )


def test_flatten_imports_combines_notebook_named_imports():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "from qmcpy import Zeta\n",
                    "from qmcpy import Alpha,Beta\n",
                    "print(Alpha)\n",
                ],
            }
        ]
    }
    source = json.dumps(notebook, indent=1).encode()

    updated, count = flatten_imports(source)

    assert count == 1
    assert json.loads(updated)["cells"][0]["source"] == [
        "from qmcpy import Alpha, Beta, Zeta\n",
        "print(Alpha)\n",
    ]
    assert flatten_imports(updated) == (updated, 0)


def test_check_mode_reports_changes_without_writing(tmp_path):
    path = tmp_path / "example.py"
    original = (_nested_import("true_measure", "Gaussian") + "\n").encode()
    path.write_bytes(original)

    assert main(["--check", str(path)]) == 1
    assert path.read_bytes() == original

    assert main([str(path)]) == 0
    assert path.read_bytes() == b"from qmcpy import Gaussian\n"
    assert main(["--check", str(path)]) == 0
