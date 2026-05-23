from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "qmc-software.yml"
OUT_PATH = ROOT / "docs" / "qmc-software.md"

helper = runpy.run_path(str(ROOT / "scripts" / "qmc_software_table.py"))

content = """# Quasi-Monte Carlo Software Packages

This page is intended to be a community-maintained resource for software related to quasi-Monte Carlo methods. Contributions are welcome.

Please submit a pull request targeting the `develop` branch with corrections or additions to the [`qmc-software.yml`](https://github.com/QMCSoftware/QMCSoftware/blob/develop/data/qmc-software.yml) data file.

If you prefer not to use GitHub pull requests, you may instead email updates to [Fred Hickernell](mailto:hickernell@illinoistech.edu).

"""

table = helper["render_qmc_software_table"](
    data_path=DATA_PATH,
    mode="web",
    return_string=True,
)

OUT_PATH.write_text(content + table + "\n", encoding="utf-8")
