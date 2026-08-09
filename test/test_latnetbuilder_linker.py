import unittest
from qmcpy.util.latnetbuilder_linker import latnetbuilder_linker

class TestLatNetBuilderLinker(unittest.TestCase):
    def test_ordinary_lattice(self):
        result_path = latnetbuilder_linker(
            lnb_dir="test/fixtures/latnetbuilder/",
            out_dir="/tmp/lnb_out/",
            fout_prefix="test"
        )
        with open(result_path) as f:
            content = f.read()
        self.assertTrue(content.startswith("# lattice\n8\n65536\n"))