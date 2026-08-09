import os
import numpy as np

# --------------------------------------------------------------------------
# Parsing outputMachine.txt 
# --------------------------------------------------------------------------

_SEP = " #"


def _split_field(line):
    """Return the value portion of a 'value #comment' line, stripped."""
    return line.split(_SEP)[0].strip()


def _parse_output_machine(text):
    """
    Parse the contents of outputMachine.txt using LatNet Builder's current
    layout (mirrors umontreal-simul/latnetbuilder's own parse_output.py).

    Returns a dict with at least: 'set_type', 'dim', 'nb_points', 'merit',
    and construction-specific fields ('gen_vector', 'modulus', 'interlacing',
    'matrices', 'nb_cols', 'nb_rows', as applicable).
    """
    lines = text.split("\n")
    header = lines[0].lower()
    merit = float(lines[1].split(": ")[1])

    if "ordinary" in header:
        dim = int(_split_field(lines[3]))
        nb_points = int(_split_field(lines[4]))
        gen_vector = [int(lines[6 + i].strip()) for i in range(dim)]
        return {
            "set_type": "Ordinary",
            "dim": dim,
            "nb_points": nb_points,
            "merit": merit,
            "gen_vector": gen_vector,
        }

    elif "polynomial" in header:
        dim = int(_split_field(lines[3]))
        if "Interlacing" in lines[4]:
            interlacing, next_line = int(_split_field(lines[4])), 7
        else:
            interlacing, next_line = 1, 5
        modulus = int(_split_field(lines[next_line]))
        n_terms = dim * interlacing
        gen_vector = [int(lines[next_line + 2 + i].strip()) for i in range(n_terms)]
        return {
            "set_type": "Polynomial",
            "dim": dim,
            "nb_points": 2 ** int(np.log2(modulus)),
            "merit": merit,
            "gen_vector": gen_vector,
            "modulus": modulus,
            "interlacing": interlacing,
        }

    elif "sobol" in header:
        dim = int(_split_field(lines[3]))
        if "Interlacing" in lines[4]:
            interlacing, next_line = int(_split_field(lines[4])), 6
        else:
            interlacing, next_line = 1, 4
        max_level = int(_split_field(lines[next_line]))
        nb_points = 2 ** max_level
        gen_vector = [[0]]
        for i in range(dim * interlacing - 1):
            gen_vector.append([int(x) for x in lines[next_line + 2 + i].split(" ")])
        return {
            "set_type": "Sobol",
            "dim": dim,
            "nb_points": nb_points,
            "merit": merit,
            "gen_vector": gen_vector,
            "interlacing": interlacing,
        }

    elif "explicit" in header:
        dim = int(_split_field(lines[3]))
        if "Interlacing" in lines[4]:
            interlacing, next_line = int(_split_field(lines[4])), 6
        else:
            interlacing, next_line = 1, 4
        nb_cols = int(_split_field(lines[next_line]))
        nb_rows = nb_cols
        nb_points = 2 ** nb_cols
        matrices_cols = []
        for c in range(dim * interlacing):
            cols = [int(x) for x in lines[next_line + 3 + c].split(" ")]
            matrices_cols.append(cols)
        return {
            "set_type": "Explicit",
            "dim": dim,
            "nb_points": nb_points,
            "merit": merit,
            "nb_cols": nb_cols,
            "nb_rows": nb_rows,
            "interlacing": interlacing,
            "matrices_cols": matrices_cols,  # each entry: nb_cols ints, one per matrix column
        }

    else:
        raise ValueError(
            "Unrecognized LatNet Builder output type in outputMachine.txt "
            f"header line: {lines[0]!r}"
        )


def _cols_to_matrix(cols, nb_rows):
    """Expand a list of column integers (bit-packed, MSB-first) into a 0/1 matrix."""
    m = len(cols)
    matrix = np.zeros((nb_rows, m), dtype=np.int64)
    for j, col in enumerate(cols):
        for i in range(nb_rows):
            matrix[i, j] = (col >> (nb_rows - 1 - i)) & 1
    return matrix


# --------------------------------------------------------------------------
# Writing LDData-format files
# --------------------------------------------------------------------------

def _write_lddata_lattice(dim, nb_points, gen_vector, filepath):
    """LDData `lattice` format (verified against LDData's README example)."""
    with open(filepath, "w") as f:
        f.write("# lattice\n")
        f.write(f"{dim}\n")
        f.write(f"{nb_points}\n")
        for z in gen_vector:
            f.write(f"{int(z)}\n")
    return filepath


def _write_lddata_dnet(dim, base, nb_rows, nb_cols, matrices, filepath):
    """
    LDData `dnet` format (best-effort mapping -- see module docstring caveat).
    Header: # dnet ; params: base b, dims s, columns k, rows r ; then s lines
    representing generating matrices (one packed row-major matrix per line).
    """
    with open(filepath, "w") as f:
        f.write("# dnet\n")
        f.write(f"{base}\n")
        f.write(f"{dim}\n")
        f.write(f"{nb_cols}\n")
        f.write(f"{nb_rows}\n")
        for M in matrices:
            f.write(" ".join(str(int(v)) for v in np.asarray(M).flatten()) + "\n")
    return filepath


# --------------------------------------------------------------------------
# Main function: latnetbuilder_linker
# --------------------------------------------------------------------------

def latnetbuilder_linker(lnb_dir="./", out_dir="./", fout_prefix="lnb4qmcpy"):
    """
    Args:
        lnb_dir (str): relative path to directory where `outputMachine.txt` is stored
            e.g. 'my_lnb/poly_lat/'
        out_dir (str): relative path to directory where output should be stored
            e.g. 'my_lnb/poly_lat_qmcpy/'
        fout_prefix (str): start of output file name.
            e.g. 'my_poly_lat_vec'

    Returns:
        str: path to an LDData-format text file that can be passed to
        QMCPy's Lattice / DigitalNetB2 in order to use the linked LatNet
        Builder generating vector/matrices,
        e.g. 'my_poly_lat_vec.lattice.8.65536.txt'

    Parses LatNet Builder's current outputMachine.txt layout (mirrors
    umontreal-simul/latnetbuilder's own python-wrapper/latnetbuilder/parse_output.py)
    and exports to the LDData plain-text format
    (https://github.com/QMCSoftware/LDData).
    """
    os.makedirs(out_dir, exist_ok=True)

    in_path = os.path.join(lnb_dir, "outputMachine.txt")
    with open(in_path) as f:
        text = f.read()

    parsed = _parse_output_machine(text)
    dim = parsed["dim"]
    nb_points = parsed["nb_points"]

    if parsed["set_type"] == "Ordinary":
        f_out = os.path.join(out_dir, f"{fout_prefix}.lattice.{dim}.{nb_points}.txt")
        return _write_lddata_lattice(dim, nb_points, parsed["gen_vector"], f_out)

    elif parsed["set_type"] == "Polynomial":
        # best-effort plattice export: base 2, dim, modulus/degree, gen vector
        f_out = os.path.join(out_dir, f"{fout_prefix}.plattice.{dim}.{nb_points}.txt")
        with open(f_out, "w") as f:
            f.write("# plattice\n")
            f.write("2\n")  # base
            f.write(f"{dim}\n")
            f.write(f"{parsed['modulus']}\n")
            for z in parsed["gen_vector"]:
                f.write(f"{int(z)}\n")
        return f_out

    elif parsed["set_type"] in ("Sobol", "Explicit"):
        nb_rows = parsed.get("nb_rows", int(np.log2(nb_points)))
        if parsed["set_type"] == "Explicit":
            matrices = [_cols_to_matrix(cols, nb_rows) for cols in parsed["matrices_cols"]]
            nb_cols = parsed["nb_cols"]
        else:
            # Sobol' direction numbers -> generating matrices (base-2 columns)
            matrices = [_cols_to_matrix(cols, nb_rows) for cols in parsed["gen_vector"]]
            nb_cols = nb_rows
        f_out = os.path.join(out_dir, f"{fout_prefix}.dnet.{dim}.{nb_rows}.{nb_cols}.txt")
        return _write_lddata_dnet(dim, 2, nb_rows, nb_cols, matrices, f_out)

    else:
        raise NotYetImplemented(
            f"latnetbuilder_linker: unsupported set_type {parsed['set_type']!r}"
        )
