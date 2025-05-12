import os
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt

#       Bring in out.res, set parameter labels and assign coulmns
mjd, res, err = np.loadtxt("out.res", unpack=True, usecols=[0, 5, 6])

figure = plt.figure()
dir_path = os.path.split(os.getcwd())[1]
plt.errorbar(mjd, res, yerr=err, marker='s', ms=1.0, color='k', ecolor='k', elinewidth=1.0, capsize=1.5, linestyle='None')
plt.xlabel('MJD', fontname = 'serif')
plt.ylabel('Timing residual [s]', fontname = 'serif')
plt.title(dir_path, fontname = 'serif')
plt.savefig(dir_path, format="pdf", bbox_inches="tight")
