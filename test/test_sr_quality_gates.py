import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestQualityGates(unittest.TestCase):

    def test_ci_develop_branch_setup_is_idempotent(self):
        workflow = (REPO_ROOT / ".github/workflows/alltests.yml").read_text()
        branch_command = next(
            line.strip()
            for line in workflow.splitlines()
            if "git branch develop refs/remotes/origin/develop" in line
        )
        commands = [shlex.split(command) for command in branch_command.split(" || ")]

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "develop"], check=True)
            subprocess.run(
                [
                    "git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit",
                    "-q", "--allow-empty", "-m", "initial",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "update-ref", "refs/remotes/origin/develop", "HEAD"],
                check=True,
            )

            self._run_shell_or(commands, repo)
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach"], check=True)
            subprocess.run(["git", "-C", str(repo), "branch", "-D", "develop"], check=True)
            self._run_shell_or(commands, repo)

            result = subprocess.run(
                ["git", "-C", str(repo), "show-ref", "--verify", "refs/heads/develop"]
            )
            self.assertEqual(result.returncode, 0)

    @staticmethod
    def _run_shell_or(commands, cwd):
        for command in commands:
            result = subprocess.run(command, cwd=cwd)
            if result.returncode == 0:
                return
        raise AssertionError("workflow branch setup command failed")

    def test_prepush_uses_read_only_strict_checks(self):
        makefile = (REPO_ROOT / "makefile").read_text()
        recipe = makefile.split("\nprepush:\n", 1)[1].split("\n\n", 1)[0]

        self.assertIn("$(MAKE) check_format", recipe)
        self.assertNotIn("$(MAKE) format", recipe)
        self.assertIn("$(MAKE) check_notebook_execution_changed STRICT=--strict", recipe)
        self.assertIn("$(MAKE) check_ref_style_changed STRICT=--strict", recipe)

    @unittest.skipIf(os.name == "nt" or shutil.which("sh") is None, "POSIX hook")
    def test_prepush_hook_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_make = Path(directory) / "make"
            fake_make.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
            fake_make.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{directory}:{env['PATH']}"

            result = subprocess.run(
                ["sh", str(REPO_ROOT / ".githooks/pre-push")],
                cwd=REPO_ROOT,
                env=env,
            )

        self.assertEqual(result.returncode, 17)


if __name__ == "__main__":
    unittest.main()
