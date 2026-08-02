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

    assert count == 2
    assert updated == (
        b"from qmcpy import Keister\n"
        b"from qmcpy import Lattice as LD\n"
        b"from qmcpy import DigitalNetB2\n"
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


def test_check_mode_reports_changes_without_writing(tmp_path):
    path = tmp_path / "example.py"
    original = (_nested_import("true_measure", "Gaussian") + "\n").encode()
    path.write_bytes(original)

    assert main(["--check", str(path)]) == 1
    assert path.read_bytes() == original

    assert main([str(path)]) == 0
    assert path.read_bytes() == b"from qmcpy import Gaussian\n"
    assert main(["--check", str(path)]) == 0
