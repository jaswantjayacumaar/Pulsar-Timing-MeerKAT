#!/usr/bin/env python

#from __future__ import print_function
#import numpy as np
import argparse
import sys, os
import subprocess
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from scipy import linalg
#from astropy import coordinates as coord
#from astropy import units as u
#from matplotlib.ticker import FormatStrFormatter
#import numpy.ma as ma
#import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid.inset_locator import inset_axes
import pulsar_glitch as pg


parser = argparse.ArgumentParser(description='Stride routine for fitting over a glitch. Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).')
parser.add_argument('-p', '--parfile', type=str, help='Parameter file', required=True)
parser.add_argument('-t', '--timfile', type=str, help='TOA file', required=True)
#parser.add_argument('-w', '--width', type=int, default=0, help='Boxcar width (days), defualt: 6*cadence')
#parser.add_argument('-s', '--step', type=int, default=0, help='Size of stride (days), defualt: 3*cadence')
#parser.add_argument('-a', '--gap', type=float, default=0, help='Threshold of gap (days), defualt: 10*cadence')
parser.add_argument('-u', '--taug', type=int, default=[200]*10, nargs='+', help="Replace GLF1 with change of spin frequency 'taug' days after the glitch respectively")
parser.add_argument('-g', '--glitches', type=int, default=[], nargs='+', help='Glitches that need to split tim file for')
parser.add_argument('-m', '--multiple', type=list, default=[], nargs='+', help='Multiple glitches that need to split tim file together for')
parser.add_argument('-r', '--recoveries', type=int, default=[0, 1], nargs='+', help='Number of recoveries to fit in models for all large glitches')
parser.add_argument('--glf2', action='store_true', help="Include GLF2 term in glitch model")
parser.add_argument('--glf0d', '--glf0d-range', type=float, default=0.8, help='The prior range for glf0d')
parser.add_argument('--sigma', '--measured-sigma', type=float, default=[100, 100, 100, 100], nargs='+', help='Minus/Plus sigma range of GLF0(instant), and Minus/Plus sigma range of GLF0(T=taug) respectively')
parser.add_argument('--split', '--gltd-split', type=float, default=[1.8, 2.3], nargs='+', help='Where to split gltd priors (in log10) for double and triple recoveries respectively')
parser.add_argument('-a', '--gap', type=float, default=0, help='Threshold of gap (days), defualt: 10*cadence')
#parser.add_argument('-g', '--glep', type=float, default=[], nargs='+', help='Glitch epochs(MJD)')
#parser.add_argument('-d', '--data', help='Stride data text file', required=True)
args = parser.parse_args()
    
# Set Pulsar class sfpsr, load info, generate truth file
sfpsr = pg.Pulsar(args.parfile, args.timfile, glf0t=args.taug)
sfpsr.delete_null()
sfpsr.print_info()
#sfpsr.generate_truth()

# Convert double recovery to tempo2 style, chop tim file, update info
#sfpsr.tidy_glitch(chop=5000)
#sfpsr.load_info()
#sfpsr.delete_null()

# Merge the best model of each glitch into the final par
#sfpsr.final_par(largeglitch=args.glitches, recoveries=args.recoveries, GLF2=args.glf2)

# Generating tim files for individual glitches, par files for different models, and run MCMC fit to find best model
for gi in args.glitches:
    sfpsr.split_tim(glitchnum=gi)
    for exp in args.recoveries:
        sfpsr.split_par(glitchnum=gi, recoveries=exp, GLF2=args.glf2)
        sfpsr.MCMC_fit(glitchnum=gi, recoveries=exp, GLF2=args.glf2, glf0drange=args.glf0d, sigma=args.sigma, gltdsplit=args.split)
        #sfpsr.post_par(glitchnum=gi, recoveries=exp)
    sfpsr.best_model(glitchnum=gi, recoveries=None, GLF2=args.glf2)

'''
for multi in args.multiple:
    start = int(multi[0])-1
    end = int(multi[-1])+1
    sfpsr.split_tim(startnum=start, endnum=end)
'''
