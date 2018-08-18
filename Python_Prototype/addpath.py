import os
import sys


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            if name[0].isalpha():
                r.append(os.path.join(root, name))
    r = list(set(r))
    return r


curr_sys_path = sys.path
print(curr_sys_path)
r = list_files(".")

for s in r:
    if not s in curr_sys_path:
        print(s)
        sys.path.append(s)

print(sys.path)
