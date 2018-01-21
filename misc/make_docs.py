"""
Assumes the following:
1. That you have installed the noxer using
    cd ~/noxer
    pip install -e .

2. That the clone of your fork of the noxer-org.github.io
repository is available in your home folder, that is in
~/noxer-org.github.io
"""

import os
import shutil

home = os.path.expanduser('~')
docs = os.path.join(home, 'noxer-org.github.io')

permanent_objects = {'.git', 'LICENSE', 'README.md', '.gitignore'}

# remove everything except for the .git folder
for f in os.listdir(docs):
    if f in permanent_objects:
        continue

    obj_loc = os.path.join(docs, f)

    if os.path.isdir(obj_loc):
        shutil.rmtree(obj_loc)
    else:
        os.remove(obj_loc)

# make html of documentation
os.system('pdoc --html --html-dir '+docs+' noxer')

# subfolder created by pdoc
subf = os.path.join(docs, 'noxer')

# move everything from the subfolder
os.system('mv ' + subf + '/* ' + docs)

# remove the subfolder
shutil.rmtree(subf)