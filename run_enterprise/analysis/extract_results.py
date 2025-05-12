#!/usr/bin/env python

import argparse
#import sys, os
#import subprocess
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import numpy as np
#import numpy.ma as ma
import pulsar_glitch as pg
#from __future__ import print_function
#from scipy.optimize import curve_fit
#from scipy import linalg
#from astropy import coordinates as coord
#from astropy import units as u
#from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid.inset_locator import inset_axes



parser = argparse.ArgumentParser(description='Stride routine for fitting over a glitch. Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).')
parser.add_argument('-p', '--parfile', type=str, help='Parameter file', required=True)
parser.add_argument('-t', '--timfile', type=str, help='TOA file', required=True)
parser.add_argument('-u', '--taug', type=int, default=[200]*100, nargs='+', help='Replace GLF1 with change of spin frequency tau_g days after the glitch respectively')
parser.add_argument('-g', '--glitches', type=int, default=[], nargs='+', help='Glitches that need to split tim file for')
parser.add_argument('-r', '--recoveries', type=int, default=[], nargs='+', help='Number of recoveries in the best model for each glitch')
parser.add_argument('--glf2', type=int, default=[], nargs='+', help='Include GLF2 term for the best model of these glitches')
args = parser.parse_args()
    
# Set Pulsar class sfpsr, load info, generate truth file
sfpsr = pg.Pulsar(args.parfile, args.timfile, glf0t=args.taug)
sfpsr.delete_null()
#sfpsr.generate_truth()

# Merge the best model of each glitch into the final par, and update par info
#sfpsr.final_par(largeglitch=args.glitches, recoveries=args.recoveries, GLF2=args.glf2)
#sfpsr.load_info(glf0t=args.taug)
#sfpsr.delete_null()

# (Run MCMC fit) and read results from sum results and final par
#sfpsr.MCMC_fit(par=args.parfile, tim=args.timfile, glitchnum=None)
sum_results = "sum_" + args.parfile.split("_")[1] + ".results"
sfpsr.noise_slt = sfpsr.read_results_file(sum_results, noise=True)
print("<<< The noise solution for pulsar {} is {} >>>".format(sfpsr.psrn, sfpsr.noise_slt))
sfpsr.par = sfpsr.post_par(par=sfpsr.sumpar, tim=sfpsr.tim, glitchnum=None)
sfpsr.read_final_par(largeglitch=args.glitches)

# Update info with par results files from best model of each glitch, and extract parameters to slt file
idx_recovery = 0
for gi in args.glitches:
    if len(args.glf2)==0:
        f2 = "a"
    elif gi in args.glf2:
        f2 = "y"
    else:
        f2 = "n"
    if len(args.recoveries) > idx_recovery:
        sfpsr.best_model(glitchnum=gi, recoveries=args.recoveries[idx_recovery], GLF2=f2)
    else:
        sfpsr.best_model(glitchnum=gi, recoveries=None, GLF2=f2)
    idx_recovery += 1
sfpsr.extract_results()
