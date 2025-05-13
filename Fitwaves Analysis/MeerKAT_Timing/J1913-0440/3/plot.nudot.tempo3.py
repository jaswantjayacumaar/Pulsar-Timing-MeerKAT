import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import sys, os
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pylab
import pylab as plt
from pylab import rc
from matplotlib import rc,rcParams
import matplotlib.gridspec as gridspec
label_size = 20
rc('text', usetex=False )
rc('font',family='serif',size= 15, weight = 'normal')
pylab.locator_params(axis='y',nbins=10)
pylab.ticklabel_format(style='plain', axis='y',scilimits=(0,0))
M_SIZE = 8

parser = argparse.ArgumentParser(description='')
parser.add_argument('-F1','--f1', help='Ascii file containing the all the nu-dot-vs-time solutions', required=True)
parser.add_argument('-mjd','--MJD', help='The MJD axis of the plot', required=True)
parser.add_argument('-p','--psr', help='The MJD axis of the plot', required=True)
args = parser.parse_args()


f1vsl = np.loadtxt(args.f1)*10**15.0
sh = np.shape(f1vsl)
dates = np.loadtxt(args.MJD)
pulsar = args.psr

ncurves = sh[0]
index = sh[1]

median_nu_dot = np.median(f1vsl, axis=0)
errorbar = np.std(f1vsl, axis=0)
print (len(median_nu_dot))

for i in range(ncurves):
	nu_dot = f1vsl[i,:]
	plt.plot(dates, nu_dot, alpha=0.4, color='skyblue')
	plt.errorbar(dates, median_nu_dot, yerr = 2.0*errorbar, fmt='-x', capsize=3.0, color='k')
plt.show()

st = int(input('How many point in the front you want to ignore, if none enter 0\n'))
en = int(input('How many point from the end you want to ignore if none enter 0\n'))
if st != 0 or en !=0:
	for i in range(ncurves):
		nu_dot = f1vsl[i,st:-en]
		plt.plot(dates[st:-en], nu_dot, alpha=0.4, color='skyblue')
		plt.errorbar(dates[st:-en], median_nu_dot[st:-en], yerr = 2.0*errorbar[st:-en], fmt='-x', capsize=3.0, color='k')
		plt.title(pulsar)
	plt.xlabel('Modified Julian Day')
	plt.ylabel(r'$\dot \nu \times 10^{-15}$ (Hz/s)')
	plt.show()

else:
	for i in range(ncurves):
		nu_dot = f1vsl[i,:]
		plt.plot(dates, nu_dot, alpha=0.1, color='skyblue')
		plt.errorbar(dates, median_nu_dot, yerr = 2.0*errorbar, fmt='-x', capsize=3.0, color='k')
		plt.title(pulsar)
	plt.xlabel('Modified Julian Day')
	plt.ylabel(r'$\dot \nu \times 10^{-15}$ (Hz/s)')
	plt.show()

np.savetxt(pulsar+'nu.dot.mjd.txt', list(zip(dates, median_nu_dot, 2.0*errorbar)), header='MJD nu_dot nu (10^-15) nu_dot err(10^-15) (2 sigma)')

