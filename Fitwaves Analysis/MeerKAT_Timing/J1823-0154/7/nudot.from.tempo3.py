import numpy as np
import argparse
import os, sys
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger
import PyPDF2



'''
import pylab
from pylab import rc
from matplotlib import rc,rcParams
import matplotlib.gridspec as gridspec
rc('text', usetex=True )
rc('font',family='serif',size= 12, weight = 'normal')
pylab.locator_params(axis='y',nbins=7)
pylab.ticklabel_format(style='plain', axis='y',scilimits=(0,0))
M_SIZE = 8
label_size = 8
'''

parser = argparse.ArgumentParser(description='Using TEMPO3 to get nu-dot variations')
parser.add_argument('-e','--ephemeris', help='Pular Ephemeris', required=True)
parser.add_argument('-m','--macrofile', help='macro file', required=True)
parser.add_argument('-b','--baryfile', help='macro file', required=True)
parser.add_argument('-n','--nsample', help='number of times the simulation need to be performed to ', required=True, type =int)
args = parser.parse_args()

macrof = args.macrofile
parfile = args.ephemeris
baryfile = args.baryfile
nsmap = args.nsample

command = '/mirror/scratch/mkeith/psrsalsa/tempo3 -perturb_toas -macro '+macrof+' -nogtk -baryssb '+parfile+' '+baryfile


def error(arr):
	sorted_arr = np.sort(arr)
	lower_idx = int((2.3/100)*len(sorted_arr))
	upper_idx = int((95.4/100)*len(sorted_arr))
	lowerlim = sorted_arr[lower_idx]
	upperlim = sorted_arr[upper_idx]
	
	return lowerlim, upperlim



i = 0
mjds =[]
nu_dots =[]
plots_name =[]
while i < nsmap:
	i +=1
	os.system(command)
	print (command)
	os.system('mv res.txt res.'+str(i)+'.txt')
	data = np.loadtxt('res.'+str(i)+'.txt')
	date = data[:,0]
	F1 = data[:,1]
	mjds.append(date)
	nu_dots.append(F1)
	os.system('rm res.'+str(i)+'.txt')
	os.system('ps2pdf pgplot.ps')
	os.system('mv pgplot.pdf iteration.'+str(i)+'.plot.pdf')
	os.system('mv parfile.par iteration.'+str(i)+'.par')
	os.system('mv timfile.tim iteration.'+str(i)+'.tim')

	plots_name.append('iteration.'+str(i)+'.plot.pdf')



merger = PyPDF2.PdfFileMerger()
for pdf in plots_name:
	input_f = PyPDF2.PdfFileReader(open(pdf,'rb'))
	merger.append((input_f))
	
merger.write('merged.plots.pdf')
merger.close()
os.system('rm iteration.*.pdf')

os.system('mkdir PARTIMFS')
os.system('mv iteration.*.par iteration.*.tim PARTIMFS')



		
mjds = np.array(mjds)
nu_dots = np.array(nu_dots)
avg_f1 = np.mean(nu_dots, axis=0)


'''
print (np.shape(nu_dots))
#low_err, up_err = error()
print (np.shape(nu_dots))
print (nu_dots)
for i in range(np.shape(nu_dots)[1]):
	F1_monte = nu_dots[:,i]
	plt.hist(F1_monte)
	plt.show()
	print (np.std(F1_monte))
	print (F1_monte)
'''

error = np.std(nu_dots, axis =0)

avg_date = np.mean(mjds, axis=0)

for k in range(np.shape(nu_dots)[0]):
	F1f = nu_dots[k,:]
	mjd = mjds[k,:]
	plt.plot(mjd, F1f*10**15, color='grey', alpha =0.4)
plt.errorbar(avg_date, avg_f1*10**15, yerr=error*10**15, color='k', linewidth=1.5, fmt='.-' )
plt.xlabel('Modified Julian Day')
plt.ylabel(r'$\dot \nu \times 10^{-15}$ (Hz/s)')

plt.show()


data = zip(avg_date, avg_f1, error)
np.savetxt('nudot.mjd.txt', list(data), comments='mjd F1 F1err')
np.savetxt('nudot.full.sample.txt', nu_dots)
np.savetxt('MJDS.txt', list(avg_date))



