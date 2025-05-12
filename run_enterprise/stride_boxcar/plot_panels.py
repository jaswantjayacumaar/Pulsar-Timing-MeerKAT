#!/usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from astropy import coordinates as coord
from astropy import units as u
import sys
from matplotlib.ticker import FormatStrFormatter
import numpy.ma as ma
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid.inset_locator import inset_axes

parser = argparse.ArgumentParser(description='Generating data files fot plot routine of fitting over a glitch.')
parser.add_argument('-p', '--parfile', help='Path to ephemeris', required=True)
parser.add_argument('-s', '--stride', help='Stride data text file', required=True)
args = parser.parse_args()
par = args.parfile
strdat = args.stride

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

pglep=np.zeros(100)
pglf0=np.zeros(100)
pglf1=np.zeros(100)
pglf2=np.zeros(100)
pglf0d=np.zeros(100)
pgltd=np.ones(100)
pglf0d2=np.zeros(100)
pgltd2=np.ones(100)
max_glitch=0

#Open par file and extract parameters
with open(par) as f:
    for line in f:
        line = line.strip()
        e=line.split()
        if e[0] == "PSRJ":
            psrn=e[1]
        if e[0].startswith("GLEP_"):
            i=int(e[0][5:])
            pglep[i-1] = float(e[1])
            max_glitch = max(i,max_glitch)
        if e[0].startswith("GLF0_"):
            i=int(e[0][5:])
            pglf0[i-1] = float(e[1])
        if e[0].startswith("GLF1_"):
            i=int(e[0][5:])
            pglf1[i-1] = float(e[1])
        if e[0].startswith("GLF2_"):
            i=int(e[0][5:])
            pglf2[i-1] = float(e[1])
        if e[0].startswith("GLTD_"):
            i=int(e[0][5:])
            pgltd[i-1] = float(e[1])
        if e[0].startswith("GLF0D_"):
            i=int(e[0][6:])
            pglf0d[i-1] = float(e[1])
        if e[0].startswith("GLTD2_"):
            i=int(e[0][6:])
            pgltd2[i-1] = float(e[1])
        if e[0].startswith("GLF0D2_"):
            i=int(e[0][7:])
            pglf0d2[i-1] = float(e[1])
        if e[0] == "F0":
            F0=float(e[1])
        if e[0] == "PB":
            PB=float(e[1])
        if e[0] == "F1":
            F1=float(e[1])
        if e[0] == "F2":
            F2=float(e[1])
        if e[0] == "START":
            start=float(e[1])
        if e[0] == "FINISH":
            finish=float(e[1])
        if e[0] == "PEPOCH":
            pepoch=float(e[1])


print("")
print("Parameters in par file")
print("F0:", F0)
print("F1:", F1)
print("F2:", F2)
print("")

for i in range(max_glitch):
    print("The {} glitch".format(i+1))
    print("Glitch epoch:", pglep[i])
    print("GLF0:", pglf0[i])
    print("GLF1:", pglf1[i])
    print("GLF2:", pglf2[i])
    print("GLF0D_1:", pglf0d[i], " - GLTD_1", pgltd[i]) 
    print("GLF0D2_1:", pglf0d2[i], " - GLTD2_1", pgltd2[i])
    print("Initial jump:", pglf0d[i]+pglf0d2[i]+pglf0[i])
    print("")


def glexp(xx,td,f0d):
    '''
    xx = time since glitch epoch
    td = decay timescale
    f0d = decay amplitude
    '''

    ee = np.zeros_like(xx)
    tau1=td*86400.0
    ee[xx>0] = f0d * np.exp(-xx[xx>0]/tau1)
    return ee


sft, f0, f0e, f1, f1e, f2, f2e, mjds, mjdf = np.loadtxt(strdat, unpack=True)
#str_mjd = sft
#str_nudot = f1
#str_err = f1e
#str_nu2dot = f2
#str_2err = f2e
dat="deltanu_{}.asc".format(psrn)
t, nu = np.loadtxt(dat,unpack=True)

#Time since period epoch
sfx = (sft-pepoch)*86400.0
x = (t-pepoch)*86400.0

#First derivative term of taylor series
sf1 = F1 * sfx
tf1 = F1 * x

#second derivative term of taylor series
#f = f0 + f1 t + 0.5 f2 t^2
sf2 = 0.5 * sfx * sfx * F2
tf2 = 0.5 * x * x * F2

dsf2 = sfx * F2
dtf2 = x * F2

sfglf0 = np.zeros_like(sft)
sfglf1 = np.zeros_like(sft)
sfglf2 = np.zeros_like(sft)
sfexp1 = np.zeros_like(sft)
sfexp2 = np.zeros_like(sft)
#str2nu = np.zeros_like(sft)
dsfglf1 = np.zeros_like(sft)
dsfglf2 = np.zeros_like(sft)
dsfexp = np.zeros_like(sft)

glf0 = np.zeros_like(t)
glf1 = np.zeros_like(t)
glf2 = np.zeros_like(t)
exp1 = np.zeros_like(t)
exp2 = np.zeros_like(t)
dglf1 = np.zeros_like(t)
dglf2 = np.zeros_like(t)
dexp = np.zeros_like(t)

gleps = []

for gi in range(len(pglep)):
    if float(pglep[gi]) != 0:
        glep = pglep[gi]
        gleps.append(pglep[gi])
        print("The {} glitch at {}".format(gi+1, glep))
        #Time since glitch epoch
        sfxx = (sft-glep)*86400.0
        xx = (t-glep)*86400.0

        #Permanent change term (constant)
        sfglf0[sfxx>0] += pglf0[gi]
        glf0[xx>0] += pglf0[gi]

        #GLF1 term
        sfglf1[sfxx>0] += sfxx[sfxx>0] * pglf1[gi]
        glf1[xx>0] += xx[xx>0] * pglf1[gi]

        #GLF2 term
        sfglf2[sfxx>0] += 0.5* (sfxx[sfxx>0]**2) * pglf2[gi]
        glf2[xx>0] += 0.5* (xx[xx>0]**2) * pglf2[gi]

        #transient terms
        sfexp1 += glexp(sfxx,pgltd[gi],pglf0d[gi])
        if pglf0d2[gi] != 0:
            sfexp2 += glexp(sfxx,pgltd2[gi],pglf0d2[gi])
        exp1 += glexp(xx,pgltd[gi],pglf0d[gi])
        if pglf0d2[gi] != 0:
            exp2 += glexp(xx,pgltd2[gi],pglf0d2[gi])

        #stride GLF1 term
        dsfglf1[sfxx>0] += pglf1[gi] 
        #Change in spin-down rate
        dglf1[xx>0] += pglf1[gi]

        #stride GLF2 term
        dsfglf2[sfxx>0] += sfxx[sfxx>0] * pglf2[gi]
        #Change in spin-down rate derivative
        dglf2[xx>0] += xx[xx>0] * pglf2[gi] 

        #subtract exp
        dsfexp += glexp(sfxx,pgltd[gi],pglf0d[gi])/(pgltd[gi]*86400) + glexp(sfxx,pgltd2[gi],pglf0d2[gi])/(pgltd2[gi]*86400)
        #subtract exp
        dexp += glexp(xx,pgltd[gi],pglf0d[gi])/(pgltd[gi]*86400) + glexp(xx,pgltd2[gi],pglf0d2[gi])/(pgltd2[gi]*86400)


plt.errorbar(sft, 1e6*(f0 - F0 - sf1 - sf2), yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
plt.show()
plt.errorbar(sft, 1e6*(f0 - F0 - sf1 - sf2 - sfglf0 - sfglf1 - sfglf2), yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
plt.show()
plt.errorbar(sft, 1e6*(f0 - F0 - sf1 - sf2 - sfglf0 - sfglf1 - sfglf2 - sfexp1 - sfexp2), yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
plt.show()

with open("panel1_{}.txt".format(psrn),"w") as file1:
    for i in range(0, len(sft)):
        file1.write('%f   %e   %e   \n'%(sft[i], 1e6*(f0[i] - F0 - sf1[i] - sf2[i]), 1e6*f0e[i]))
    file1.close()

with open("panel2_{}.txt".format(psrn),"w") as file2:
    for i in range(0, len(sft)):
        file2.write('%f   %e   %e   \n'%(sft[i], 1e6*(f0[i] - F0 - sf1[i] - sf2[i] - sfglf0[i] - sfglf1[i] - sfglf2[i]), 1e6*f0e[i]))
    file2.close()

with open("panel3_{}.txt".format(psrn),"w") as file3:
    for i in range(0, len(sft)):
        file3.write('%f   %e   %e   \n'%(sft[i], 1e6*(f0[i] - F0 - sf1[i] - sf2[i] - sfglf0[i] - sfglf1[i] - sfglf2[i] - sfexp1[i] - sfexp2[i]), 1e6*f0e[i]))
    file3.close()


t, no_nudot, nudot, nudot_model = np.loadtxt("nudot_{}.asc".format(psrn), unpack=True)
#this_mjd, this_dnu, this_dnu_err = np.loadtxt("mjd_dnu_dnuerr_nof2.txt", unpack=True)
panel1_mjd, panel1_nu, panel1_err = np.loadtxt("panel1_{}.txt".format(psrn), unpack=True)
panel2_mjd, panel2_nu, panel2_err = np.loadtxt("panel2_{}.txt".format(psrn), unpack=True)
panel3_mjd, panel3_nu, panel3_err = np.loadtxt("panel3_{}.txt".format(psrn), unpack=True)


glep = pglep[0]
mask_len = []
for i in range(len(t)):
    for gi in range(len(gleps)):
        if t[i] <= pglep[gi] < t[i+1]:
            mask_len.append(i) # or i+1
# mask the data at glitch

#frequency evolution with second derivative and spin-down change subtracted
numod = nu - tf2 # why only f2? deltanu=nu-F0-f1?
mc = ma.array(numod)
mc[mask_len] = ma.masked # mc: mask data at glep

#md = ma.array(no_nudot+(dglf1+dglf2-dexp)/1e-15)
md = ma.array(nudot_model-(F1+dtf2)*1e15)  # In principle the same after convert the units
md[mask_len] = ma.masked # md: mask data at glep


if all(f0d==0 for f0d in pglf0d): # remove the 3rd panel
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7.5, 10.6)) #10.6

    plt.subplot(411)
    plt.plot(t-glep,1e6*mc, 'k-', zorder=2)
    plt.errorbar(panel1_mjd - glep, panel1_nu, yerr=panel1_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1)
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=15)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])

    plt.subplot(412)
    plt.plot(t-glep,1e6*(mc-glf0-glf1-glf2), 'k-', zorder=2)
    plt.errorbar(panel2_mjd - glep, panel2_nu, yerr=panel2_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=15, labelpad=15)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])

    plt.subplot(413)
    plt.plot(t-glep, md/1e5, 'k-', zorder=2)
    plt.errorbar(sft - glep, (f1-F1-dsf2)/1e-10, yerr=f1e/1e-10, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    plt.ylabel(r'$\dot{\nu}$ ($10^{-10}$ Hz s$^{-1}$)', fontsize=15)
    plt.xlabel("Days since glitch epoch", fontsize=15)
    plt.subplots_adjust(wspace=0, hspace=0.002)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])

    plt.subplot(414)
    plt.plot(t-glep, md/1e5-(dglf1+dglf2-dexp)/1e-10, 'k-', zorder=2)
    plt.errorbar(sft - glep, (f1-F1-dsf2-dsfglf1-dsfglf2+dsfexp)/1e-10, yerr=f1e/1e-10, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    plt.ylabel(r'$\dot{\nu}_{\mathrm{ng}}$ ($10^{-10}$ Hz s$^{-1}$)', fontsize=15)
    plt.xlabel("Days since glitch epoch", fontsize=15)
    plt.subplots_adjust(wspace=0, hspace=0.002)

    plt.tight_layout()
    plt.savefig("nu_nudot_gp_{}.pdf".format(psrn), format='pdf', dpi=400)
    plt.show()


else:
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(7.5, 12.8)) #10.6

    plt.subplot(511)
    plt.plot(t-glep,1e6*mc, 'k-', zorder=2)
    plt.errorbar(panel1_mjd - glep, panel1_nu, yerr=panel1_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1)
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=15)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])

    plt.subplot(512)
    plt.plot(t-glep,1e6*(mc-glf0-glf1), 'k-', zorder=2)
    plt.errorbar(panel2_mjd - glep, panel2_nu, yerr=panel2_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=15, labelpad=15)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])

    #If exists recovery
    plt.subplot(513)
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=15)
    plt.plot(t-glep,1e6*(mc-glf0-glf1-exp1-exp2), 'k-', zorder=2)
    plt.errorbar(panel3_mjd - glep, panel3_nu, yerr=panel3_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])

    plt.subplot(514)
    plt.plot(t-glep, md/1e5, 'k-', zorder=2)
    plt.errorbar(sft - glep, (f1-F1-dsf2)/1e-10, yerr=f1e/1e-10, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    plt.ylabel(r'$\dot{\nu}$ ($10^{-10}$ Hz s$^{-1}$)', fontsize=15)
    plt.xlabel("Days since glitch epoch", fontsize=15)
    plt.subplots_adjust(wspace=0, hspace=0.002)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])

    plt.subplot(515)
    plt.plot(t-glep, md/1e5-(dglf1+dglf2-dexp)/1e-10, 'k-', zorder=2)
    plt.errorbar(sft - glep, (f1-F1-dsf2-dsfglf1-dsfglf2+dsfexp)/1e-10, yerr=f1e/1e-10, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        plt.axvline(gls-glep, color='k', linestyle='dotted', alpha=0.3, linewidth=2)
    plt.ylabel(r'$\dot{\nu}_{\mathrm{ng}}$ ($10^{-10}$ Hz s$^{-1}$)', fontsize=15)
    plt.xlabel("Days since glitch epoch", fontsize=15)
    plt.subplots_adjust(wspace=0, hspace=0.002)

    plt.tight_layout()
    plt.savefig("nu_nudot_gp_{}.pdf".format(psrn), format='pdf', dpi=400)
    plt.show()
