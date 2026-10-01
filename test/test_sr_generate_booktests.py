import tempfile
import unittest
from pathlib import Path

from test.booktests import generate_test


class TestGenerateBooktests(unittest.TestCase):

    def test_skips_scratch_notebooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            demos = Path(tmp) / "demos" / "portfolio"
            demos.mkdir(parents=True)
            (demos / "_RUN_tmp.ipynb").write_text("{}")
            (demos / "real_demo.ipynb").write_text("{}")
            out_dir = Path(tmp) / "booktests"
            out_dir.mkdir()

            generated = generate_test.generate_missing_tests(
                demos_dir=str(demos.parent), output_dir=out_dir
            )

            self.assertEqual([Path(p).name for p in generated], ["tb_real_demo.py"])
            self.assertFalse((out_dir / "tb__RUN_tmp.py").exists())


if __name__ == "__main__":
    unittest.main()
