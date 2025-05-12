#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import argparse
import sys, os
import subprocess
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import linalg
from astropy import coordinates as coord
from astropy import units as u
from matplotlib.ticker import FormatStrFormatter
import numpy.ma as ma
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import pulsar_glitch as pg


parser = argparse.ArgumentParser(description='Stride routine for fitting over a glitch. Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).')
parser.add_argument('-p', '--parfile', type=str, help='Parameter file', required=True)
parser.add_argument('-t', '--timfile', type=str, help='TOA file', required=True)
parser.add_argument('-w', '--width', type=int, default=0, help='Boxcar width (days), defualt: 6*cadence')
parser.add_argument('-s', '--step', type=int, default=0, help='Size of stride (days), defualt: 3*cadence')
parser.add_argument('-a', '--gap', type=float, default=0, help='Threshold of gap (days), defualt: 10*cadence')
parser.add_argument('-u', '--taug', type=int, default=[200]*100, nargs='+', help="Replace GLF1 with change of spin frequency 'taug' days after the glitch respectively")
parser.add_argument('-g', '--glitches', type=int, default=[], nargs='+', help='Glitches that need to split tim file for')
parser.add_argument('-m', '--multiple', type=list, default=[], nargs='+', help='Multiple glitches that need to split tim file together for')
parser.add_argument('-r', '--recoveries', type=int, default=[], nargs='+', help='Number of recoveries in the best model for each glitch')
parser.add_argument('--glf2', action='store_true', help="Include GLF2 term in glitch model")
#parser.add_argument('-g', '--glep', type=float, default=[], nargs='+', help='Glitch epochs(MJD)')
#parser.add_argument('-d', '--data', help='Stride data text file', required=True)
args = parser.parse_args()
    
# Set Pulsar class sfpsr, load info, generate truth file
sfpsr = pg.Pulsar(args.parfile, args.timfile, glf0t=args.taug)
sfpsr.delete_null()
#sfpsr.generate_truth()

# Convert double recovery to tempo2 style, chop tim file, update info
#sfpsr.tidy_glitch(chop=5000)
#sfpsr.load_info()
#sfpsr.delete_null()

# Merge the best model of each glitch into the final par and update par info
sfpsr.final_par(largeglitch=args.glitches, recoveries=args.recoveries, GLF2=args.glf2)
sfpsr.load_info()
sfpsr.delete_null()
sfpsr.print_info()

# Calculate data for pulsar plots
sfpsr.noglitch_par()
sfpsr.pp_create_files()
rx, ry, re, y_dat, freqs, pwrs = sfpsr.pp_calculate_data()

# Load data for pulsar plots
rx2, ry2, re2 = np.loadtxt("out2_{}.res".format(sfpsr.psrn), usecols=(0, 5, 6), unpack=True)
t, y = np.loadtxt("ifunc_{}.asc".format(sfpsr.psrn), usecols=(1, 2), unpack=True)
t1, yf2 = np.loadtxt("deltanu_{}.asc".format(sfpsr.psrn), usecols=(0, 1), unpack=True)
t2, yd, yd_model, yd2 = np.loadtxt("nudot_{}.asc".format(sfpsr.psrn), usecols=(0, 1, 2, 3), unpack=True)

# Test
print("Shapes of data")
print(np.shape(t), np.shape(t1), np.shape(t2), np.shape(y), np.shape(yf2), np.shape(yd), np.shape(yd_model), np.shape(yd2))
print(np.shape(rx), np.shape(ry), np.shape(re), np.shape(rx2), np.shape(ry2), np.shape(re2), np.shape(y_dat))
print(np.shape(freqs), np.shape(pwrs))
# Check

# Make pulsar plots
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(311)
plt.errorbar(rx, ry2, yerr=re2, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.title("PSR "+sfpsr.psrn)
plt.xlabel("MJD")
plt.ylabel("residual (s)")
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel("Residual (turns)")
ax = fig.add_subplot(312)
plt.plot(t, y, color='green')
plt.errorbar(rx, ry, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.xlabel("MJD")
plt.ylabel("residual (s)")
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel("Residual (turns)")
ax = fig.add_subplot(313)
plt.plot(t,y-y,color='green')
plt.errorbar(rx, ry-y_dat, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.xlabel("MJD")
plt.ylabel("Residual - Model (s)")
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel("Residual - Model (turns)")
plt.savefig("residuals_{}.pdf".format(sfpsr.psrn))
plt.close()

plt.figure(figsize=(16,9))
plt.plot(t, yd, color='blue')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
plt.title("PSR "+sfpsr.psrn)
plt.xlabel("MJD")
plt.ylabel("$\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)")
plt.savefig("nudot_{}.pdf".format(sfpsr.psrn))
plt.close()

fig = plt.figure(figsize=(16,9))
fig.suptitle("PSR "+sfpsr.psrn)
ax = fig.add_subplot(321)
plt.errorbar(rx, ry2, yerr=re2, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.ylabel("residual (s)")
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
ax3 = ax.twiny()
ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004) # MJD 53005 = 2004.01.01
ax3.xaxis.tick_top()
ax3.set_xlabel("Year")
ax3.xaxis.set_tick_params(direction='inout', labeltop=True)
ax.xaxis.set_tick_params(labelbottom=False, direction='in')
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel("Residual (turns)")
ax = fig.add_subplot(323)
plt.plot(t, y, color='green')
plt.errorbar(rx, ry, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.ylabel("residual (s)")
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
ax.xaxis.set_tick_params(labelbottom=False, direction='in')
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel("Residual (turns)")
ax = fig.add_subplot(325)
plt.plot(t, y-y, color='green')
plt.errorbar(rx, ry-y_dat, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.xlabel("MJD")
plt.ylabel("Residual - Model (s)")
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
ax.xaxis.set_tick_params(labelbottom=True, direction='inout')    
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel("Residual - Model (turns)")
ax = fig.add_subplot(322)
plt.plot(t, 1e6*yf2, color='orange')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
plt.xlabel("MJD")
plt.ylabel("$\\Delta{\\nu}$ ($\mathrm{\mu}$Hz)")
ax.yaxis.set_label_position("right")
ax.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')
ax3 = ax.twiny()
ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004)
ax3.xaxis.tick_top()
ax3.set_xlabel("Year")
ax = fig.add_subplot(324)
plt.plot(t, yd_model, color='lightblue', ls='--')
plt.plot(t, yd2, color='blue')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
plt.xlabel("MJD")
plt.ylabel("$\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)")
ax.yaxis.set_label_position("right")
ax.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')
ax = fig.add_subplot(326)
plt.plot(t, yd, color='blue')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle="--", color='purple', alpha=0.7)
plt.xlabel("MJD")
plt.ylabel("$\\Delta\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)")
ax.yaxis.set_label_position("right")
ax.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')
plt.subplots_adjust(hspace=0, wspace=0.15)
plt.figtext(x=0.47, y=0.94, s="$P$:{:.0f} ms".format(1000.0/sfpsr.F0), horizontalalignment='center')
plt.figtext(x=0.53, y=0.94, s="$\\dot{{P}}$:{:.0g}".format(-sfpsr.F1/(sfpsr.F0**2)), horizontalalignment='center')
if sfpsr.PB == None:
    pass
elif sfpsr.PB > 0:
    plt.figtext(x=0.59, y=0.94, s="$P_B$:{:.1g}".format(sfpsr.PB), horizontalalignment='center')
plt.savefig("combined_{}.pdf".format(sfpsr.psrn))

plt.figure()
plt.figtext(x=0.47, y=0.94, s="$P$:{:.0f} ms".format(1000.0/sfpsr.F0), horizontalalignment='center')
plt.figtext(x=0.53, y=0.94, s="$\\dot{{P}}$:{:.0g}".format(-sfpsr.F1/(sfpsr.F0**2)), horizontalalignment='center')
if sfpsr.PB == None:
    pass
elif sfpsr.PB > 0:
    plt.figtext(x=0.59, y=0.94, s="$P_B$:{:.1g}".format(sfpsr.PB), horizontalalignment='center')
plt.loglog(freqs, pwrs)
plt.title("PSR "+sfpsr.psrn)
plt.xlabel("Freq (yr^-1)")
plt.xlabel("Power (???)")
plt.savefig("pwrspec_{}.pdf".format(sfpsr.psrn))
plt.close()

# Do stride fitting, calculate stride fitting results, save in panels
sfpsr.sf_main(width=args.width, step=args.step, F1e=5e-15)
sfpsr.sf_calculate_data()

# Load analytic model, and stride data from panels
t, nu = np.loadtxt("deltanu_{}.asc".format(sfpsr.psrn), unpack=True)
t, nudot, nudot_mod, nudot_sum = np.loadtxt("nudot_{}.asc".format(sfpsr.psrn), unpack=True)
p1_mjd, p1_nu, p1_err = np.loadtxt("panel1_{}.txt".format(sfpsr.psrn), unpack=True)
p2_mjd, p2_nu, p2_err = np.loadtxt("panel2_{}.txt".format(sfpsr.psrn), unpack=True)
p3_mjd, p3_nu, p3_err = np.loadtxt("panel3_{}.txt".format(sfpsr.psrn), unpack=True)
p4_mjd, p4_nudot, p4_err = np.loadtxt("panel4_{}.txt".format(sfpsr.psrn), unpack=True)
p5_mjd, p5_nudot, p5_err = np.loadtxt("panel5_{}.txt".format(sfpsr.psrn), unpack=True)

# Calculating analytic model terms
x = pg.mjd2sec(t, sfpsr.pepoch)
tf1, tf2, tdf2 = sfpsr.psr_taylor_terms(x)
tglf0, tglf1, tglf2, texp1, texp2, texp3, tdglf1, tdglf2, tdexp = sfpsr.glitch_terms(t)

# Calculating stride fitting terms
sfx = pg.mjd2sec(p1_mjd, sfpsr.pepoch)
sftf1, sftf2, sftdf2 = sfpsr.psr_taylor_terms(sfx)
sftglf0, sftglf1, sftglf2, sftexp1, sftexp2, sftexp3, sftdglf1, sftdglf2, sftdexp = sfpsr.glitch_terms(p1_mjd)

nu_panel = nu - tf2 # why only f2? deltanu=nu-F0-f1?
manu = sfpsr.mask_glep(t, nu_panel)
#nudot_panel = nudot + (tdglf1 + tdglf2 - tdexp)*1e15 # In principle the same after convert the units
nudot_p2 = nudot_sum - sfpsr.F1*1e15  
manudot2 = sfpsr.mask_glep(t, nudot_p2)
nudot_panel = nudot_sum - (sfpsr.F1 + tdf2)*1e15  
manudot = sfpsr.mask_glep(t, nudot_panel)
glep = sfpsr.pglep[0]
gleps = sfpsr.pglep

#test
t_mask, t_inverse = sfpsr.toa_gap(t, gap=args.gap)

# Plot the spin frequency and spin-down rate residuals panels
if all(f0d==0 for f0d in sfpsr.pglf0d): # remove the 3rd panel
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 10.6)) #10.6
    plt.subplot(411)
    #plt.scatter(sfpsr.toaseries-glep, [0]*len(sfpsr.toaseries), c='b', marker=2, linewidth=0.5, alpha=0.2, label='TOAs')
    plt.plot(t_mask-glep,1e6*manu, 'k-', zorder=2, label='Analytic model')
    plt.plot(t_inverse-glep,1e6*manu, 'g-', zorder=2, label='Extrapolation', alpha=0.8)
    plt.errorbar(p1_mjd - glep, p1_nu, yerr=p1_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, label='Stirde fit')
    #plt.ylabel(r'$\nu-\nu_{sd}$ ($\mu$Hz)', fontsize=15)
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=15)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    for toa in sfpsr.toaseries:
        plt.axvline(toa-glep, ymax=0.05, color='b', linestyle='dashed', alpha=0.2, linewidth=0.5)
    #plt.ylim(ymin=-1)
    #plt.xticks(sfpsr.toaseries-glep, color='b', alpha=0.4)
    plt.legend(loc='upper left')
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    plt.subplot(412)
    plt.plot(t_mask-glep,1e6*(manu-tglf0-tglf1-tglf2), 'k-', zorder=2)
    plt.plot(t_inverse-glep,1e6*(manu-tglf0-tglf1-tglf2), 'g-', zorder=2, alpha=0.8)
    plt.errorbar(p2_mjd - glep, p2_nu, yerr=p2_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.ylabel(r'$\nu-\nu_{sd}-\nu_{gp}$ ($\mu$Hz)', fontsize=15, labelpad=15)
    plt.ylabel(r'$\delta \nu-\nu_{gp}$ ($\mu$Hz)', fontsize=15, labelpad=15)
    frame = plt.gca()
    #frame.axes.xaxis.set_ticklabels([])
    plt.subplot(413)
    plt.plot(t_mask-glep, manudot2/1e5, 'k-', zorder=2)
    plt.plot(t_inverse-glep, manudot2/1e5, 'g-', zorder=2, alpha=0.8)
    plt.errorbar(p4_mjd - glep, p4_nudot, yerr=p4_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.ylabel(r'$\dot{\nu}-\dot{\nu_{sd}}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    plt.ylabel(r'$\delta \dot{\nu}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    #plt.ylim(5*np.min(np.min(manudot2/1e5), 0), 5*np.max(np.max(manudot2/1e5), 0))
    plt.subplots_adjust(wspace=0, hspace=0.002)
    frame = plt.gca()
    #frame.axes.xaxis.set_ticklabels([])
    plt.subplot(414)
    plt.plot(t_mask-glep, manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10, 'k-', zorder=2)
    plt.plot(t_inverse-glep, manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10, 'g-', zorder=2, alpha=0.8)
    plt.errorbar(p5_mjd - glep, p5_nudot, yerr=p5_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.ylabel(r'$\dot{\nu}-\dot{\nu_{sd}}-\dot{\nu_{gp}}-\dot{\nu_{gt}}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    plt.ylabel(r'$\delta \dot{\nu}-\dot{\nu_{g}}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    plt.xlabel("Days since first glitch epoch", fontsize=15)
    #plt.ylim(5*np.min(np.min(manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10), 0), 5*np.max(np.max(manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10), 0))
    plt.subplots_adjust(wspace=0, hspace=0.002)
    plt.tight_layout()
    plt.savefig("nu_nudot_gp_{}.pdf".format(sfpsr.psrn), format='pdf', dpi=400)
    plt.show()

else:
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(7.5, 12.8)) #10.6
    plt.subplot(511)
    #plt.scatter(sfpsr.toaseries-glep, [-1]*len(sfpsr.toaseries), c='b', marker='|', linewidth=0.5, alpha=0.4, label='TOAs')
    #plt.annotate('ToAs', xy=(sfpsr.toaseries-glep, [0]*len(sfpsr.toaseries)), xycoords='data', annotation_clip=False, alpha=0.4)
    plt.plot(t_mask-glep,1e6*manu, 'k-', zorder=2, label='Analytic model')
    plt.plot(t_inverse-glep,1e6*manu, 'g-', zorder=2, label='Extrapolation', alpha=0.8)
    plt.errorbar(p1_mjd - glep, p1_nu, yerr=p1_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, label='Stirde fit')
    #plt.ylabel(r'$\nu-\nu_{sd}$ ($\mu$Hz)', fontsize=15)
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=15)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    for toa in sfpsr.toaseries:
        plt.axvline(toa-glep, ymax=0.05, color='b', linestyle='dashed', alpha=0.2, linewidth=0.5)
    #plt.ylim(ymin=-1)
    #plt.xticks(sfpsr.toaseries-glep, color='b', alpha=0.4)
    plt.legend(loc='upper left')
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    plt.subplot(512)
    plt.plot(t_mask-glep,1e6*(manu-tglf0-tglf1-tglf2), 'k-', zorder=2)
    plt.plot(t_inverse-glep,1e6*(manu-tglf0-tglf1-tglf2), 'g-', zorder=2, alpha=0.8)
    plt.errorbar(p2_mjd - glep, p2_nu, yerr=p2_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.ylabel(r'$\nu-\nu_{sd}-\nu_{gp}$ ($\mu$Hz)', fontsize=15, labelpad=15)
    plt.ylabel(r'$\delta \nu-\nu_{gp}$ ($\mu$Hz)', fontsize=15, labelpad=15)
    frame = plt.gca()
    #frame.axes.xaxis.set_ticklabels([])
    #If exists recovery
    plt.subplot(513)
    #plt.ylabel(r'$\nu-\nu_{sd}-\nu_{gp}-\nu_{gt}$ ($\mu$Hz)', fontsize=15)
    plt.ylabel(r'$\delta \nu-\nu_{g}$ ($\mu$Hz)', fontsize=15)
    plt.plot(t_mask-glep,1e6*(manu-tglf0-tglf1-tglf2-texp1-texp2-texp3), 'k-', zorder=2)
    plt.plot(t_inverse-glep,1e6*(manu-tglf0-tglf1-tglf2-texp1-texp2-texp3), 'g-', zorder=2, alpha=0.8)
    plt.errorbar(p3_mjd - glep, p3_nu, yerr=p3_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    frame = plt.gca()
    #frame.axes.xaxis.set_ticklabels([])
    plt.subplot(514)
    plt.plot(t_mask-glep, manudot2/1e5, 'k-', zorder=2)
    plt.plot(t_inverse-glep, manudot2/1e5, 'g-', zorder=2, alpha=0.8)
    plt.errorbar(p4_mjd - glep, p4_nudot, yerr=p4_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.ylabel(r'$\dot{\nu}-\dot{\nu_{sd}}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    plt.ylabel(r'$\delta \dot{\nu}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    #plt.ylim(5*np.min(np.min(manudot2/1e5), 0), 5*np.max(np.max(manudot2/1e5), 0))
    plt.subplots_adjust(wspace=0, hspace=0.002)
    frame = plt.gca()
    #frame.axes.xaxis.set_ticklabels([])
    plt.subplot(515)
    plt.plot(t_mask-glep, manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10, 'k-', zorder=2)
    plt.plot(t_inverse-glep, manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10, 'g-', zorder=2, alpha=0.8)
    plt.errorbar(p5_mjd - glep, p5_nudot, yerr=p5_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.ylabel(r'$\dot{\nu}-\dot{\nu_{sd}}-\dot{\nu_{gp}}-\dot{\nu_{gt}}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    plt.ylabel(r'$\delta \dot{\nu}-\dot{\nu_{g}}$ ($10^{-10}$Hz$^{2}$)', fontsize=15)
    plt.xlabel("Days since first glitch epoch", fontsize=15)
    #plt.ylim(5*np.min(np.min(manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10), 0), 5*np.max(np.max(manudot/1e5-(tdglf1+tdglf2-tdexp)/1e-10), 0))
    plt.subplots_adjust(wspace=0, hspace=0.002)
    plt.tight_layout()
    plt.savefig("nu_nudot_gp_{}.pdf".format(sfpsr.psrn), format='pdf', dpi=400)
    plt.show()

