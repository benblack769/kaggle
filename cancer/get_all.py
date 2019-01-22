import subprocess
import os

lines = [l.strip() for l in open("get_files_large.txt").readlines()]
for l in lines:
    if os.path.exists("examples/"+l):
        print("skipping: "+l)
    else:
        cmd = "scp  -i  ~/.ssh/newdesktop/wdesktop benblack@192.168.0.12:fun_projs/kaggle/cancer/train/{} examples/".format(l)
        subprocess.check_call(cmd,shell=True)
