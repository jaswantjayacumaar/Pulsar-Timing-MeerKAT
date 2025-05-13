import numpy as np
import os, sys
import glob

parfile = glob.glob('iteration.*.par')
timfile = glob.glob('iteration.*.tim')

command = 'tempo2 -f ', parfile, timfile

os.system(command)

print(command)



