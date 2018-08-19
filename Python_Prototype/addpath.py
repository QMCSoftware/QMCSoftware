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

this_dir = os.path.dirname(os.path.abspath(__file__))
r = list_files(this_dir)

for s in r:
    if not s in curr_sys_path:
        #print(s)
        sys.path.append(s)

print(list(set(sys.path)-set(curr_sys_path)))
