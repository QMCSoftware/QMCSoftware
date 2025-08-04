import os
import re
import zipfile

# === Configuration ===
input_file = '../ntbktests/all_notebook_tests.py'    # path to your input file
output_dir = 'booktests'          # directory to write individual test files


# === Read the original file ===
with open(input_file, 'r') as f:
    lines = f.readlines()

# === Prepare header (lines 1-8) and footer (lines 31-32) ===
header = lines[:4] + ['\n']           # keep lines 1–8 plus a blank line
bottom = ['\n'] + lines[159:161]        # a blank line plus lines 31–32

# === Find all @testbook blocks ===
tests = []
for idx, line in enumerate(lines):
    if line.strip().startswith('@testbook'):
        decorator = line
        def_line  = lines[idx + 1]
        pass_line = lines[idx + 2]
        tests.append((decorator, def_line, pass_line))

# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Create individual test files ===
for decorator, def_line, pass_line in tests:
    # Extract notebook base name from decorator, e.g. 'ray_tracing' from "...ray_tracing.ipynb"
    m = re.search(r".*/([^/]+)\.ipynb'", decorator)
    if not m:
        continue
    name = m.group(1)
    filename = f"tb_{name}.py"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, 'w') as tf:
        # Write header, then the single test, then footer
        tf.writelines(header)
        tf.write(decorator)
        tf.write(def_line)
        tf.write(pass_line)
        tf.writelines(bottom)

