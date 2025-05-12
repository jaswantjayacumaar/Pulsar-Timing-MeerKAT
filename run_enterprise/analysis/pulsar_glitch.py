#!/usr/bin/env python
'''Pulsar glitch module for processing glitch data.
    Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).'''

import time
import argparse
import sys, os
import subprocess
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import numpy as np
import numpy.ma as ma
import pandas as pd
#from __future__ import print_function
from uncertainties import ufloat
#from scipy.optimize import curve_fit
#from scipy import linalg
#from astropy import coordinates as coord
#from astropy import units as u
#from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid.inset_locator import inset_axes

def mjd2sec(t, epoch):
    '''Convert MJD t into unit of seconds: x = time since epoch in seconds'''
    x = (t-epoch)*86400.0
    return x

def lin_fit(t, mjd, yoffs):
    '''Linear fit for discrete yoffs as a function of mjd at time t'''
    if t < mjd[0]:
        return yoffs[0]
    elif t > mjd[-1]:
        return yoffs[-1]
    else:
        for i in range(len(mjd)):
            if t > mjd[i] and t < mjd[i+1]:
                x1, x2 = mjd[i], mjd[i+1]
                y1, y2 = yoffs[i], yoffs[i+1]
                x=(t-x1)/(x2-x1)
                return (y2-y1)*x + y1

def recovery_name(recoveries):
    '''Check the value of recoveries, set to 0 if exceed 3, and return the corresponding modelname'''
    if isinstance(recoveries, int) and 0 <= recoveries <= 3:
        pass
    else:
        print("Invalid recoveries value {} is set to 0".format(recoveries))
        recoveries = 0
    if recoveries==0:
        modelname = "f"
    elif recoveries==1:
        modelname = "r"
    elif recoveries==2:
        modelname = "d"
    elif recoveries==3:
        modelname = "t"
    return recoveries, modelname

def std_error(parameter, error):
    '''Set the parameter and its non-zero standard deviation in ufloat format'''
    if error == 0 or error is np.nan or error is None:
        return parameter
    else:
        return ufloat(parameter, error)


class Pulsar:
    '''A pulsar class consists basic info of pulsar'''
    def __init__(self, parfile, timfile, tidy=False, glf0t="GLTD"):
        '''Initialize pulsar class with corresponding par file and tim file'''
        self.par, self.tim = parfile, timfile
        self.psrn, self.ra, self.dec, self.PB = [None]*4
        self.F0, self.F1, self.F2 = [None]*3
        self.F0std, self.F1std, self.F2std = [None]*3
        self.start, self.finish, self.pepoch = [None]*3
        self.tidypar, self.testpar, self.truth = [None]*3
        self.epochfile, self.datafile = [None]*2
        self.spin_slt, self.gli_slt, self.noise_slt = [{}]*3
        self.load_info(tidy=tidy, glf0t=glf0t)

    def load_info(self, tidy=False, glf0t="GLTD"):
        '''Load basic info of pulsar class from par file and tim file'''
        if tidy==False or self.tidypar==None:
            loadpar = self.par
        else:
            loadpar = self.tidypar
        spin, position, toa, red, white, glitch, recovery, summary = self.load_par(loadpar, tidy=tidy, glf0t=glf0t)
        self.psrn, self.F0, self.F1, self.F2, self.F0std, self.F1std, self.F2std = spin
        self.ra, self.dec, self.pmra, self.pmdec, self.PX, self.DM = position
        self.start, self.finish, self.pepoch = toa
        self.redamp, self.redgam, self.redc = red
        self.ef_jbafb, self.ef_jbdfb, self.eq_jbafb, self.eq_jbdfb = white
        self.pglep, self.pglf0, self.pglf1, self.pglf2, self.pglf0ins, self.pglf0tg = glitch
        self.pglf0d, self.pgltd, self.pglf0d2, self.pgltd2, self.pglf0d3, self.pgltd3 = recovery
        self.max_glitch, self.waittime, self.taug, self.numofexp = summary
        self.minmjds, self.maxmjds, self.toanum, self.toaseries, self.toaspan, self.cadence = self.load_tim(self.tim)
        self.E_dot, self.tau_c, self.B_sur, self.brake = self.calculate_derived()

    def load_par(self, par, tidy=False, glf0t="GLTD"):
        '''Load basic info of par file'''
        max_glitch = 0
        pglep, waittime, numofexp = np.zeros(100), np.zeros(100), np.zeros(100)
        pglf0, pglf1, pglf2 = np.zeros(100), np.zeros(100), np.zeros(100)
        pglf0ins, pglf0tg = np.zeros(100), np.zeros(100)
        pglf0d, pglf0d2, pglf0d3 = np.zeros(100), np.zeros(100), np.zeros(100)
        pgltd, pgltd2, pgltd3 = np.ones(100), np.ones(100), np.ones(100)
        taug = 200*np.ones(100)
        redamp, redgam, redc = [None]*3
        ef_jbafb, ef_jbdfb, eq_jbafb, eq_jbdfb = [None]*4
        psrn, F0, F1, F2, F0std, F1std, F2std = [None]*7
        ra, dec, pmra, pmdec, PX, DM = [None]*6
        pepoch, start, finish = [None]*3
        with open(par) as f1:
            for line in f1:
                line = line.strip()
                e = line.split()
                if e[0] == "PSRJ":
                    psrn = e[1]
                elif e[0] == "RAJ":
                    ra = e[1]
                elif e[0] == "DECJ":
                    dec = e[1]
                elif e[0] == "F0":
                    F0 = float(e[1])
                    if len(e) > 3:
                        F0std = float(e[3])
                elif e[0] == "F1":
                    F1 = float(e[1])
                    if len(e) > 3:
                        F1std = float(e[3])
                elif e[0] == "F2":
                    F2 = float(e[1])
                    if len(e) > 3:
                        F2std = float(e[3])
                elif e[0] == "PEPOCH":
                    pepoch = float(e[1])
                elif e[0] == "DM":
                    DM = float(e[1])
                elif e[0] == "PMRA":
                    pmra = float(e[1])
                elif e[0] == "PMDEC":
                    pmdec = float(e[1])
                elif e[0] == "PX":
                    PX = float(e[1])
                elif e[0] == "START":
                    start = float(e[1])
                elif e[0] == "FINISH":
                    finish = float(e[1])
                elif e[0] == "PEPOCH":
                    pepoch = float(e[1])
                elif e[0] == "TNRedAmp":
                    redamp = float(e[1])
                elif e[0] == "TNRedGam":
                    redgam = float(e[1])
                elif e[0] == "TNEF":
                    if e[2] == "jbafb":
                        ef_jbafb = float(e[-1])
                    elif e[2] == "jbdfb":
                        ef_jbdfb = float(e[-1])
                elif e[0] == "TNEQ":
                    if e[2] == "jbafb":
                        eq_jbafb = float(e[-1])
                    elif e[2] == "jbdfb":
                        eq_jbdfb = float(e[-1])
                elif e[0].startswith("GLEP_"):
                    i = int(e[0][5:])
                    pglep[i-1] = float(e[1])
                    max_glitch = max(i, max_glitch)
                elif e[0].startswith("GLF0_"):
                    i = int(e[0][5:])
                    pglf0[i-1] = float(e[1])
                elif e[0].startswith("GLF1_"):
                    i = int(e[0][5:])
                    pglf1[i-1] = float(e[1])
                elif e[0].startswith("GLF2_"):
                    i = int(e[0][5:])
                    pglf2[i-1] = float(e[1])
                elif e[0].startswith("GLTD_"):
                    i = int(e[0][5:])
                    pgltd[i-1] = float(e[1])
                elif e[0].startswith("GLF0D_"):
                    i = int(e[0][6:])
                    pglf0d[i-1] = float(e[1])
                    numofexp[i-1] = max(1, numofexp[i-1])
                elif e[0].startswith("GLTD2_"):
                    i = int(e[0][6:])
                    pgltd2[i-1] = float(e[1])
                elif e[0].startswith("GLF0D2_"):
                    i = int(e[0][7:])
                    pglf0d2[i-1] = float(e[1])
                    numofexp[i-1] = max(2, numofexp[i-1])
                elif e[0].startswith("GLTD3_"):
                    i = int(e[0][6:])
                    pgltd3[i-1] = float(e[1])
                elif e[0].startswith("GLF0D3_"):
                    i = int(e[0][7:])
                    pglf0d3[i-1] = float(e[1])
                    numofexp[i-1] = max(3, numofexp[i-1])
                elif e[0].startswith("GLF0(T="):
                    i = int(e[0].split("_")[-1])
                    numb = e[0].split("=")[1]
                    taug[i-1] = int(numb.split(")")[0])
        for i in range(max_glitch):
            if glf0t is "GLTD":
                if pgltd[i] != 1:
                    taug[i] = int(pgltd[i])
                if pgltd2[i] != 1:
                    taug[i] = int(min(taug[i], pgltd2[i]))
                if pgltd3[i] != 1:
                    taug[i] = int(min(taug[i], pgltd3[i]))
            elif i<len(glf0t):
                taug[i] = int(glf0t[i])
            else:
                pass
            pglf0ins[i] = pglf0[i]
            pglf0tg[i] = pglf1[i] * taug[i] + 0.5 * pglf2[i] * taug[i]**2
            if pglf0d[i] != 0:
                pglf0ins[i] += pglf0d[i]
                pglf0tg[i] += pglf0d[i]*(np.exp(-taug[i]/pgltd[i])-1)
            if pglf0d2[i] != 0:
                pglf0ins[i] += pglf0d2[i]
                pglf0tg[i] += pglf0d2[i]*(np.exp(-taug[i]/pgltd2[i])-1)
            if pglf0d3[i] != 0:
                pglf0ins[i] += pglf0d3[i]
                pglf0tg[i] += pglf0d3[i]*(np.exp(-taug[i]/pgltd3[i])-1)
            if i>0:
                waittime[i] = pglep[i]-pglep[i-1]
        spin_para = psrn, F0, F1, F2, F0std, F1std, F2std
        position_para = ra, dec, pmra, pmdec, PX, DM
        toa_para = start, finish, pepoch
        red_para = redamp, redgam, redc
        white_para = ef_jbafb, ef_jbdfb, eq_jbafb, eq_jbdfb
        glitch_para = pglep, pglf0, pglf1, pglf2, pglf0ins, pglf0tg
        recovery_para = pglf0d, pgltd, pglf0d2, pgltd2, pglf0d3, pgltd3
        glitch_summary = max_glitch, waittime, taug, numofexp
        return spin_para, position_para, toa_para, red_para, white_para, glitch_para, recovery_para, glitch_summary

    def load_tim(self, tim):
        '''Load basic info of tim file'''
        minmjds = 1e6
        maxmjds = 0
        toanum = 0
        toaseries = []
        with open(tim) as f:
            for line in f:
                e = line.split()
                if len(e) > 2 and e[0] != "C" and "-pn" in e:
                    toaseries.append(float(e[2]))
                    minmjds = min(minmjds, float(e[2]))
                    maxmjds = max(maxmjds, float(e[2]))
                    toanum += 1
        toaseries.sort()
        toaseries = np.array(toaseries)
        toaspan = maxmjds-minmjds
        cadence = toaspan/(toanum-1)
        return minmjds, maxmjds, toanum, toaseries, toaspan, cadence

    def toa_gap(self, x, gap=0):
        '''Find huge gap in TOAs and create mask array for x in MJD'''
        self.toainterval = []
        for i, toai in enumerate(self.toaseries):
            if i == 0:
                self.toainterval.append(0)
            else:
                self.toainterval.append(toai-self.toaseries[i-1])
        self.toainterval = np.array(self.toainterval)
        if gap<=0:
            gap = 10*self.cadence
        print("Threshold of gap:", gap)
        maski = x<0
        counter = 1
        for i, value in enumerate(self.toainterval):
            if value >= gap:
                print("The No.%i gap (%f) in TOA is from %f to %f"%(counter, value, self.toaseries[i-1], self.toaseries[i]))
                counter += 1
                mask1 = self.toaseries[i-1] < x
                mask2 = x < self.toaseries[i]
                maski += mask1 * mask2
        x_mask = ma.array(x, mask=maski)
        x_inverse = ma.array(x, mask=~maski)
        return x_mask, x_inverse

    def delete_null(self):
        ''' Delete empty entries in glitch parameters'''
        self.pglep = self.pglep[:self.max_glitch]
        self.pglf0 = self.pglf0[:self.max_glitch]
        self.pglf1 = self.pglf1[:self.max_glitch]
        self.pglf2 = self.pglf2[:self.max_glitch]
        self.pglf0d = self.pglf0d[:self.max_glitch]
        self.pgltd = self.pgltd[:self.max_glitch]
        self.pglf0d2 = self.pglf0d2[:self.max_glitch]
        self.pgltd2 = self.pgltd2[:self.max_glitch]
        self.pglf0d3 = self.pglf0d3[:self.max_glitch]
        self.pgltd3 = self.pgltd3[:self.max_glitch]
        self.pglf0ins = self.pglf0ins[:self.max_glitch]
        self.pglf0tg = self.pglf0tg[:self.max_glitch]
        self.waittime = self.waittime[:self.max_glitch]
        self.taug = self.taug[:self.max_glitch]
        self.numofexp = self.numofexp[:self.max_glitch]

    def generate_truth(self):
        ''' Generate truth file'''
        self.truth = "trh_"+self.psrn+".txt"
        with open(self.truth, "w") as f:
            for gi in range(self.max_glitch):
                idx = gi + 1
                f.write("GLEP_%i   %f\n"%(idx, self.pglep[gi]))
                if self.pglf0[gi] != 0:
                    f.write("GLF0_%i   %e\n"%(idx, self.pglf0[gi]))
                if self.pglf1[gi] != 0:
                    f.write("GLF1_%i   %e\n"%(idx, self.pglf1[gi]))
                if self.pglf2[gi] != 0:
                    f.write("GLF2_%i   %e\n"%(idx, self.pglf2[gi]))
                if self.pglf0d[gi] != 0:
                    f.write("GLF0D_%i   %e\n"%(idx, self.pglf0d[gi]))
                if self.pgltd[gi] != 0:
                    f.write("GLTD_%i   %f\n"%(idx, self.pgltd[gi]))
                if self.pglf0d2[gi] != 0:
                    f.write("GLF0D2_%i   %e\n"%(idx, self.pglf0d2[gi]))
                if self.pgltd2[gi] != 0:
                    f.write("GLTD2_%i   %f\n"%(idx, self.pgltd2[gi]))
                if self.pglf0d3[gi] != 0:
                    f.write("GLF0D3_%i   %e\n"%(idx, self.pglf0d3[gi]))
                if self.pgltd3[gi] != 0:
                    f.write("GLTD3_%i   %f\n"%(idx, self.pgltd3[gi]))
                if self.pglf0[gi] != 0 and self.pglf1[gi] != 0:
                    #glf0_i = glf0 + glf0d + glf0d2 + glf0d3
                    #glf0_T = glf1*t200*86400+glf0d*(np.exp(-t200/gltd)-1)+glf0d2*(np.exp(-t200/gltd2)-1)+glf0d3*(np.exp(-t200/gltd3)-1)
                    f.write("GLF0(instant)_%i   %e\n"%(idx, self.pglf0ins[gi]))
                    f.write("GLF0(T=%d)_%i   %e\n"%(self.taug[gi], idx, self.pglf0tg[gi]))
            if all(p is not None for p in [self.redamp, self.redgam]):
                #alpha = redgam
                #P0 = ((redamp**2)/(12*np.pi**2))*(fc**(-alpha))
                f.write("TNRedAmp   %f\n"%self.redamp)
                f.write("TNRedGam   %f\n"%self.redgam)

    def tidy_glitch(self, chop=None):
        ''' Sort the recovery terms according to their time scale.
            Chop tim file and only keep TOAs from chop days before first glitch to chop days after the last glitch'''
        glitches = {}
        parlines = []
        timlines = []
        with open(self.par) as f1:
            for line in f1:
                if line.startswith("GL"):
                    e = line.split()
                    pp = e[0].split("_")
                    i = int(pp[1])
                    param = pp[0]
                    if not i in glitches:
                        glitches[i] = {"turns":0}
                    if param == "GLEP":
                        glitches[i]["epoch"] = float(e[1])
                    #if param == "GLPH":
                        #glitches[i]["turns"] = round(float(e[1]))
                        #e[1] = "{}".format(float(e[1]) - glitches[i]["turns"])
                    glitches[i][param] = " ".join(e[1:])
                else:
                    parlines.append(line)
        #for ig in glitches:
            #print("glitch[{}] epoch {} turns {}".format(ig,glitches[ig]["epoch"],glitches[ig]["turns"]))
        gg = sorted(glitches, key=lambda x: glitches[x]["epoch"])
        for ig in gg:
            if "GLTD" in glitches[ig]:
                if "GLTD2" in glitches[ig]: # Swap 1st and 2nd recoveries if the 1st longer than 2nd
                    if glitches[ig]["GLTD"] > glitches[ig]["GLTD2"]:
                        glitches[ig]["GLF0D"], glitches[ig]["GLF0D2"] = glitches[ig]["GLF0D2"], glitches[ig]["GLF0D"]
                        glitches[ig]["GLTD"], glitches[ig]["GLTD2"] = glitches[ig]["GLTD2"], glitches[ig]["GLTD"]
                if "GLTD3" in glitches[ig]: # Swap 1st and 3rd recoveries if the 1st longer than 3rd
                    if glitches[ig]["GLTD"] > glitches[ig]["GLTD3"]:
                        glitches[ig]["GLF0D"], glitches[ig]["GLF0D3"] = glitches[ig]["GLF0D3"], glitches[ig]["GLF0D"]
                        glitches[ig]["GLTD"], glitches[ig]["GLTD3"] = glitches[ig]["GLTD3"], glitches[ig]["GLTD"]
                    if "GLTD2" in glitches[ig] and glitches[ig]["GLTD2"] > glitches[ig]["GLTD3"]:
                        glitches[ig]["GLF0D2"], glitches[ig]["GLF0D3"] = glitches[ig]["GLF0D3"], glitches[ig]["GLF0D2"]
                        glitches[ig]["GLTD2"], glitches[ig]["GLTD3"] = glitches[ig]["GLTD3"], glitches[ig]["GLTD2"]
        self.tidypar = "tdy_"+self.par.split("_", 1)[1]
        with open(self.tidypar,"w") as f1:
            f1.writelines(parlines)
            for ig in gg:
                for param in glitches[ig]:
                    if param in ["epoch","turns"]:
                        continue
                    f1.write("{}_{} {}\n".format(param,glitches[ig]["newid"],glitches[ig][param]))
        with open(self.tim) as f2:
            for line in f2:
                e = line.split()
                if "-pn" in e:
                    epoch = float(e[2])
                    ii = e.index("-pn")
                    pn = int(e[ii+1])
                    for ig in gg:
                        if epoch > glitches[ig]["epoch"]:
                            pn -= glitches[ig]["turns"]
                    newline = " ".join(e[:ii])+" -pn {} ".format(pn)+(" ".join(e[ii+2:]))
                    if isinstance(chop, int):
                        if self.pglep[0]-chop <= epoch <= self.pglep[-1]+chop and "-be" in e:
                            timlines.append(" "+newline+"\n")
                    else:
                        timlines.append(" "+newline+"\n")
                elif len(e)>3:
                    epoch = float(e[3])
                    if isinstance(chop, int):
                        if self.pglep[0]-chop <= epoch <= self.pglep[-1]+chop and "-be" in e:
                            timlines.append(line)
                    else:
                        timlines.append(" "+newline+"\n")
                else:
                    timlines.append(line)
        if isinstance(chop, int):
            self.tim = "chp_"+self.psrn+".tim"
        with open(self.tim,"w") as f2:
            f2.writelines(timlines)

    def split_tim(self, glitchnum=None, startnum=0, endnum=-1):
        '''Split the tim file and only keep the ToAs relevant for glitch No.glitchnum.
           Alternatively chop tthe tim file between the glitch No.startnum and glitch No.endnum, 0 is the start and -1 is the end of tim'''
        if startnum<0:
            startnum = 0
        if endnum>self.max_glitch:
            endnum = -1
        if isinstance(glitchnum, int):
            if 0 < glitchnum <= self.max_glitch:
                pass
            else:
                glitchnum = 1
            startnum = glitchnum - 1
            endnum = glitchnum + 1
            if endnum > self.max_glitch:
                endnum = -1
        if startnum == 0:
            startmjds = self.minmjds
        else:
            startmjds = self.pglep[startnum-1]
        if endnum == -1:
            endmjds = self.maxmjds
        else:
            endmjds = self.pglep[endnum-1]
        if isinstance(glitchnum, int):
            startmjds = min(startmjds, self.pglep[glitchnum-1]-1000)
            endmjds = max(endmjds, self.pglep[glitchnum-1]+1000)
        timlines = []
        with open(self.tim) as f1:
            for line in f1:
                e = line.split()
                if "-pn" in e:
                    epoch = float(e[2])
                    ii = e.index("-pn")
                    pn = int(e[ii+1])
                    newline = " ".join(e[:ii])+" -pn {} ".format(pn)+(" ".join(e[ii+2:]))
                    if startmjds <= epoch <= endmjds and "-be" in e:
                        timlines.append(" "+newline+"\n")
                elif len(e)>3:
                    epoch = float(e[3])
                    if startmjds <= epoch <= endmjds and "-be" in e:
                        timlines.append(line)
                else:
                    timlines.append(line)
        if isinstance(glitchnum, int):
            self.splittim = "chp_"+self.psrn+"_"+str(glitchnum)+".tim"
        elif startnum==0 and endnum==-1:
            self.splittim = "chp_"+self.psrn+"_all.tim"
        else:
            self.splittim = "chp_"+self.psrn+"_"+str(startnum)+":"+str(endnum)+".tim"
        print("Generating split tim {} for glitch {}".format(self.splittim, self.splittim.split(".")[-2].split("_")[-1]))
        with open(self.splittim,"w") as f2:
            f2.writelines(timlines)
        return self.splittim

    def split_par(self, splittim=None, glitchnum=None, recoveries=0, GLF2=False, small_glitches=True, pre_gli=True):
        '''Generate new split par file of different models for MCMC fit, move PEPOCH to the center with epoch centre.
           Glitchnum and recoveries specify the number of recovery terms in model for the glitch to be fit.
           Turn on tempo2 fitting for small glitches when small_glitches is True.
           Include the best fit results for the previous glitch in split par when pre_gli is True'''
        if splittim is None:
            splittim = self.splittim
        if isinstance(glitchnum, int) and 0 < glitchnum <= self.max_glitch:
            pass
        else:
            glitchnum = int(self.splittim.split("_")[-1].split(".")[0].split(":")[0])
        recoveries, modelname = recovery_name(recoveries)
        f2 = "n"
        if GLF2 is True:
            f2 = "y"
        self.splitpar = "bst_"+self.psrn+"_"+str(glitchnum)+modelname+f2+".par"
        subprocess.call(["tempo2", "-epoch", "centre", "-nofit", "-f", self.par, splittim, "-outpar", self.splitpar])
        print("Generating split par {} for glitch {} model {} (GLF2: {})".format(self.splitpar, glitchnum, modelname, f2))
        splitminmjd, splitmaxmjd = self.load_tim(splittim)[:2]
        sml_gli = np.where(np.logical_and(self.pglep>splitminmjd, self.pglep<splitmaxmjd))[0]+1
        sml_gli = np.delete(sml_gli, np.where(sml_gli == glitchnum))
        print("The small glitches included in splittim are:", sml_gli)
        pre_tim = "chp_"+self.psrn+"_"+str(glitchnum-1)+".tim"
        if pre_gli is True and glitchnum>1 and os.path.exists(pre_tim):
            pre_par = self.best_model(glitchnum=glitchnum-1)
            print("Found the best post results {} for previous glitch No.{}".format(pre_par.rsplit("/",maxsplit=1)[-1], glitchnum-1))
        else:
            pre_gli = False
        parlines = []
        with open(self.splitpar) as f1:
            for line in f1:
                e = line.split()
                if any(line.startswith(ls) for ls in ["GLEP", "GLF0_", "GLF1_", "GLF2_", "GLF0("]):
                    if line.startswith("GLF2_{} ".format(str(glitchnum))):
                        GLF2=False
                    if small_glitches is True:
                        if line.startswith("GLF") and int(e[0].split("_")[-1]) in sml_gli:
                        # Turn on tempo2 fitting of these parameters for small glitches included in this range
                            if len(e) >= 3:
                                newline = e[0]+"   "+e[1]+"   "+"1"+"   "+e[-1]+"\n"
                                parlines.append(newline)
                            else:
                                newline = e[0]+"   "+e[1]+"   "+"1"+"\n"
                                parlines.append(newline)
                            continue                         
                    parlines.append(line)
                elif line.startswith("TN") or line.startswith("GLPH"):
                    continue
                elif any(line.startswith(ls) for ls in ["PX", "PM", "DM ", "JUMP"]):
                    parlines.append(line) # Maybe also turn on?
                elif any(line.startswith(ls) for ls in ["RA", "DEC", "F0", "F1", "F2"]):
                    # Turn on tempo2 fitting of these parameters
                    if len(e) >= 3:
                        newline = e[0]+"   "+e[1]+"   "+"1"+"   "+e[-1]+"\n"
                        parlines.append(newline)
                    else:
                        newline = e[0]+"   "+e[1]+"   "+"1"+"\n"
                        parlines.append(newline)
                elif line.startswith("GLF0D_") or line.startswith("GLTD_"):
                    num = int(e[0].split("_")[-1])
                    if num==glitchnum and recoveries < 1:
                        pass
                    else:
                        parlines.append(line)
                elif line.startswith("GLF0D2_") or line.startswith("GLTD2_"):
                    num = int(e[0].split("_")[-1])
                    if num==glitchnum and recoveries < 2:
                        pass
                    else:
                        parlines.append(line)
                elif line.startswith("GLF0D3_") or line.startswith("GLTD3_"):
                    num = int(e[0].split("_")[-1])
                    if num==glitchnum and recoveries < 3:
                        pass
                    else:
                        parlines.append(line)
                else:
                    parlines.append(line)
        expterms = self.numofexp[glitchnum-1]
        for expnum in range(3):
            if recoveries > expterms:
                if expterms == expnum:
                    if expnum == 0:
                        newline = "GLF0D_"+str(glitchnum)+"   "+"0"+"\n"
                        parlines.append(newline)
                        newline = "GLTD_"+str(glitchnum)+"   "+str(50*(4**expnum))+"\n"
                        parlines.append(newline)
                    else:
                        newline = "GLF0D{}_".format(str(expnum+1))+str(glitchnum)+"   "+"0"+"\n"
                        parlines.append(newline)
                        newline = "GLTD{}_".format(str(expnum+1))+str(glitchnum)+"   "+str(50*(4**expnum))+"\n"
                        parlines.append(newline)
                    expterms += 1
        if GLF2 is True:
            newline = "GLF2_{}".format(str(glitchnum))+"   "+"0"+"\n"
            parlines.append(newline)
        if pre_gli is True:
            pre_para = lambda x : x.split()[0].startswith("GL") and int(x.split()[0].split("_")[-1])==glitchnum-1
            parlines[:] = [line for line in parlines if not pre_para(line)]
            with open(pre_par) as f2:
                for line in f2:
                    e = line.split()
                    if any(e[0].startswith(ls) for ls in ["GLEP", "GLF0_", "GLF1_", "GLF2_", "GLF0D", "GLTD"]) and int(e[0].split("_")[-1])==glitchnum-1:                    
                        parlines.append(line)
        with open(self.splitpar, "w") as f3:
            f3.writelines(parlines)
        return self.splitpar

    def MCMC_fit(self, par=None, tim=None, glitchnum=1, solver="multinest", recoveries=0, GLF2=False, sigma=[100, 100, 100, 100], gleprange=2, gltdrange=[1, 3], gltdsplit=[2.0, 2.3], glf0drange=0.8, glf0range=0.8, glf1range=0.8, glf2range=10, red=[-16, -8], tspan=1.1, thread=16, N=8000, nwalkers=128, nlive=500, rootdir="/nvme1/yliu/yangliu/"):
        '''Call enterprise to do MCMC fit for model selection for par and tim on glitch glitchnum.
           Use options to specify the prior ranges and enterprise settings.
           The directory for enterpresie folder is rootdir'''
        if not os.path.exists(os.path.join(rootdir,"run_enterprise/run_enterprise.py")):
            rootdir = "/nvme1/yliu/yangliu/"
        runentprise = os.path.join(rootdir,"run_enterprise/run_enterprise.py")
        if par is None:
            par = self.splitpar
        if tim is None:
            tim = self.splittim
        if isinstance(glitchnum, int) and 0 < glitchnum <= self.max_glitch:
            pass
        elif glitchnum is None:
            pass
        else:
            glitchnum = int(self.splittim.split("_")[-1].split(".")[0].split(":")[0])
        recoveries, modelname = recovery_name(recoveries)
        f2 = "n"
        if GLF2 is True:
            f2 = "y"
        else:
            glf2range = 0
        if recoveries == 3:
            gltdsplit[0] = 1.8
        command = ["mpirun", "-c", str(thread), runentprise, "--auto-add"]
        command.extend([par, tim])
        if glitchnum is not None:
            plotname = str(glitchnum)+modelname+f2
            command.extend(["--glitches", str(glitchnum)])
        else:
            plotname = "sum"
        dirname = self.psrn+"_"+plotname
        outputfile = "opt_"+dirname+".txt"
        taug = [str(int(tg)) for tg in self.taug]
        plotchain = "--plot-chain"
        command.extend(["--outdir", dirname, "--plotname", plotname])
        if solver=="emcee":
            command.extend(["-N", str(N), "--nwalkers", str(nwalkers)])
        elif solver=="dynesty":
            plotchain = "--dynesty-plots"
            command.extend(["-nlive", str(nlive)])
        else:
            solver = "multinest"
        print("######")
        print("Start MCMC fitting")
        print("Tau_g is:", self.taug)
        print("Directory name is:", dirname)
        print("Par file for MCMC fit:", par)
        print("Tim file for MCMC fit:", tim)
        print("######")
        solver = "--"+solver
        command.extend([solver, plotchain, "--plot-derived"]) 
        command.extend(["-j", "--red-prior-log", "--Ared-min", str(red[0]), "-A", str(red[1])])
        command.extend(["--tspan-mult", str(tspan), "--glitch-alt-f0", "--glitch-alt-f0t"])
        command.extend(taug)
        command.extend(["--measured-prior", "--measured-sigma", str(sigma[0]), str(sigma[1]), str(sigma[2]), str(sigma[3])])
        command.extend(["--glitch-epoch-range", str(gleprange), "--glitch-td-min", str(gltdrange[0]), "--glitch-td-max", str(gltdrange[1])])
        command.extend(["--glitch-td-split", str(gltdsplit[0]), str(gltdsplit[1]), "--glitch-f0d-range", str(glf0drange)])
        command.extend(["--glitch-f0-range", str(glf0range), "--glitch-f1-range", str(glf1range), "--glitch-f2-range", str(glf2range)]) 
        print("### MCMC Command is:", command)
        opt = open(outputfile, "w")
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            opt.write(line.decode("utf-8"))

    def post_par(self, par=None, tim=None, glitchnum=1):
        '''Create final par file for glitch glitchnum's model based on post fit par file'''
        if par is None:
            par = self.splitpar
        if tim is None:
            tim = self.splittim
        if isinstance(glitchnum, int) and 0 < glitchnum <= self.max_glitch:
            pass
        elif glitchnum is None:
            pass
        else:
            glitchnum = int(self.splittim.split("_")[-1].split(".")[0].split(":")[0])
        modelname = par.split("_")[-1].split(".")[0][-2]
        f2 = par.split("_")[-1].split(".")[0][-1]
        postpar = par+".post"
        self.fnlpar = "fnl_"+par.split("_", 1)[1]
        subprocess.call(["tempo2", "-f", postpar, tim, "-outpar", self.fnlpar])
        if glitchnum is not None:
            print("Generating final par {} for glitch {} model {} (GLF2: {})".format(self.fnlpar, glitchnum, modelname, f2))
        else:
            print("Generating final par {} for glitch summary".format(self.fnlpar))    
        return self.fnlpar

    def best_model(self, glitchnum=1, recoveries=None, GLF2="a", max_like=False):
        '''Find the best model par for glitch glitchnum based on lnZ.
           Or specify the best model manually with the number of recoveries'''
        if isinstance(glitchnum, int) and 0 < glitchnum <= self.max_glitch:
            pass
        else:
            glitchnum = int(self.splittim.split("_")[-1].split(".")[0].split(":")[0])
        if max_like is True:
            results_value = "max-likelihood"
        else:
            results_value = "mean"
        prefix = "bst_"
        rootdir = os.getcwd()
        if GLF2 == "n":
            f2fix = ["n"]
        elif GLF2 == "y":
            f2fix = ["y"]
        else:
            f2fix = ["n", "y"]
        glnum = "_"+str(glitchnum)
        postfix = ".par.post"
        results = ".par.results"
        maxlnZ = -np.inf
        bestmodelpar = None
        bestmodel = None
        if isinstance(recoveries, int) and 0 <= recoveries <= 3:
            recoveries, model = recovery_name(recoveries)
            for f2 in f2fix:
                postname = prefix+self.psrn+glnum+model+f2+postfix
                postpar = os.path.join(rootdir, postname)
                if os.path.exists(postpar):
                    print("Found results for glitch {} model {} (GLF2: {})".format(glitchnum, model, f2))
                    ev_file = os.path.join(rootdir,"{}{}{}{}".format(self.psrn, glnum, model, f2), "pmn-stats.dat")
                    if os.path.exists(ev_file):
                        with open(ev_file) as f:
                            line = f.readline()
                            line = f.readline()
                            #lnev = float(line.split()[5])
                            lnev = ufloat(float(line.split()[5]), float(line.split()[7]))
                            print("Evidence of glitch {} model {} (GLF2: {}) is {}".format(glitchnum, model, f2, lnev))
                            if lnev > maxlnZ:
                                if lnev < maxlnZ+2.5:
                                    print(">>> Evidence Warning <<<")
                                    print("<<< Glitch {} model {} (GLF2: {}) has similar evidence >>>".format(glitchnum, model, f2))
                                else:
                                    maxlnZ = lnev
                                    bestmodelpar = postpar
                                    bestmodel = model
                                    bestf2 = f2
                    else:
                        print(">>> Missing evidence file {} <<<".format(ev_file))
        else:
            for model in ["f", "r", "d", "t"]:
                for f2 in f2fix:
                    postname = prefix+self.psrn+glnum+model+f2+postfix
                    postpar = os.path.join(rootdir, postname)
                    if os.path.exists(postpar):
                        print("Found results for glitch {} model {} (GLF2: {})".format(glitchnum, model, f2))
                        ev_file = os.path.join(rootdir,"{}{}{}{}".format(self.psrn, glnum, model, f2), "pmn-stats.dat")
                        if os.path.exists(ev_file):
                            with open(ev_file) as f:
                                line = f.readline()
                                line = f.readline()
                                #lnev = float(line.split()[5])
                                lnev = ufloat(float(line.split()[5]), float(line.split()[7]))
                                print("Evidence of glitch {} model {} (GLF2: {}) is {}".format(glitchnum, model, f2, lnev))
                                if lnev > maxlnZ:
                                    if lnev < maxlnZ+2.5:
                                        print(">>> Evidence Warning <<<")
                                        print("<<< Glitch {} model {} (GLF2: {}) has similar evidence >>>".format(glitchnum, model, f2))
                                    else:
                                        maxlnZ = lnev
                                        bestmodelpar = postpar
                                        bestmodel = model
                                        bestf2 = f2
                        else:
                            print(">>> Missing evidence file {} <<<".format(ev_file))
        print("### Best model par is:", bestmodelpar)
        if bestmodelpar is None:
            print(">>> Missing post par for best model <<<")
        else:
            print("<<< The best model for glitch {} is model {} (GLF2: {}) >>>".format(glitchnum, bestmodel, bestf2))
            rslt_file = bestmodelpar.replace("post","results")
            print("<<< Reading {} results from {} for glitch {} >>>".format(results_value, rslt_file, glitchnum))
            self.gli_slt[str(glitchnum)] = self.read_results_file(rslt_file, max_like=max_like)
            print("<<< The solution for glitch {} is {} >>>".format(glitchnum, self.gli_slt[str(glitchnum)]))
        return bestmodelpar

    def final_par(self, largeglitch=[1], recoveries=[None]*10, GLF2=[], max_like=False):
        '''Merge glitch parameters in the best model of each glitches to create final par for the pulsar'''
        self.sumpar = "sum_"+self.psrn+".par"
        parlines = []
        with open(self.par) as f1:
            for line in f1:
                if line.startswith("GL"):
                    e = line.split()
                    glnum = int(e[0].split("_")[-1])
                    if glnum not in largeglitch:
                        if line.startswith("GLEP"):
                            newline = e[0]+"   "+e[1]+"\n"
                        elif line.startswith("GLPH"):
                            newline = "#"+line
                        else: # Turn on tempo2 fitting of all other glitch parameters for small glitches
                            if len(e) >= 3:
                                newline = e[0]+"   "+e[1]+"   "+"1"+"   "+e[-1]+"\n"
                            else:
                                newline = e[0]+"   "+e[1]+"   "+"1"+"\n"
                        parlines.append(newline)
                elif line.startswith("TN"):
                    continue
                elif any(line.startswith(ls) for ls in ["RA", "DEC", "PX", "PM", "DM ", "F0", "F1", "F2"]):
                    parlines.append(line)
                else:
                    parlines.append(line)
        idx = 0
        for glnum in largeglitch:
            if len(GLF2)==0:
                f2 = "a"
            elif glnum in GLF2:
                f2 = "y"
            else:
                f2 = "n"
            if idx<len(recoveries):
                bmpar = self.best_model(glitchnum=glnum, recoveries=recoveries[idx], GLF2=f2)
            else:
                bmpar = self.best_model(glitchnum=glnum, recoveries=None, GLF2=f2)
            with open(bmpar) as f:
                for line in f:
                    if any(line.startswith(ls) for ls in ["GLEP", "GLF0_", "GLF1_", "GLF2_", "GLF0D", "GLTD"]):
                        e = line.split()
                        if glnum == int(e[0].split("_")[-1]):
                            parlines.append(line)
            idx += 1
        with open(self.sumpar, "w") as f3:
            f3.writelines(parlines)
        self.tim = self.split_tim()
        self.MCMC_fit(par=self.sumpar, tim=self.tim, glitchnum=None)
        self.sum_results = self.sumpar+".results"
        if max_like is True:
            results_value = "max-likelihood"
        else:
            results_value = "mean"
        print("<<< Reading white noise and red noise {} results from {} >>>".format(results_value, self.sum_results))
        self.noise_slt = self.read_results_file(self.sum_results, max_like=max_like, noise=True)
        print("<<< The noise solution for pulsar {} is {} >>>".format(self.psrn, self.noise_slt))
        self.par = self.post_par(par=self.sumpar, tim=self.tim, glitchnum=None)
        self.read_final_par(largeglitch=largeglitch)
        return self.par

    def read_results_file(self, res_file, max_like=False, noise=False):
        '''Read uncertainty of glitch parameters from results file, and update pulsar class info.
           Use max likelihood value when max_like is True, otherwise use mean value for parameters.
           Only read glitch parameters when noise is False, otherwise read white and red noise '''
        results={}
        self.pglepstd, self.pglf0std, self.pglf1std, self.pglf2std = np.full(self.max_glitch, np.nan), np.full(self.max_glitch, np.nan), np.full(self.max_glitch, np.nan), np.full(self.max_glitch, np.nan)
        self.pglf0insstd, self.pglf0tgstd = np.full(self.max_glitch, np.nan),np.full(self.max_glitch, np.nan)
        self.pglf0dstd, self.pglf0d2std, self.pglf0d3std = np.full(self.max_glitch, np.nan), np.full(self.max_glitch, np.nan), np.full(self.max_glitch, np.nan)
        self.pgltdstd, self.pgltd2std, self.pgltd3std = np.full(self.max_glitch, np.nan), np.full(self.max_glitch, np.nan), np.full(self.max_glitch, np.nan)
        self.redampstd, self.redgamstd = [None]*2
        self.ef_jbafbstd, self.ef_jbdfbstd, self.eq_jbafbstd, self.eq_jbdfbstd = [None]*4
        if max_like is True:
            value_idx = 1
        else:
            value_idx = 2
        with open(res_file) as f:
            f.readline()
            for line in f:
                e=line.strip().rsplit(maxsplit=8)
                if noise is False:
                    if e[0].startswith("GL"):
                        glnum = int(e[0].split("_")[-1])
                        glkey = e[0].split("_")[0]
                        if e[0].startswith("GLEP_"):
                            self.pglep[glnum-1] = float(e[value_idx])
                            self.pglepstd[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF0_"):
                            self.pglf0[glnum-1] = float(e[value_idx])
                            self.pglf0std[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF1_"):
                            self.pglf1[glnum-1] = float(e[value_idx])
                            self.pglf1std[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF2_"):
                            self.pglf2[glnum-1] = float(e[value_idx])
                            self.pglf2std[glnum-1] = float(e[3])
                        elif e[0].startswith("GLTD_"):
                            self.pgltd[glnum-1] = float(e[value_idx])
                            self.pgltdstd[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF0D_"):
                            self.pglf0d[glnum-1] = float(e[value_idx])
                            self.pglf0dstd[glnum-1] = float(e[3])
                        elif e[0].startswith("GLTD2_"):
                            self.pgltd2[glnum-1] = float(e[value_idx])
                            self.pgltd2std[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF0D2_"):
                            self.pglf0d2[glnum-1] = float(e[value_idx])
                            self.pglf0d2std[glnum-1] = float(e[3])
                        elif e[0].startswith("GLTD3_"):
                            self.pgltd3[glnum-1] = float(e[value_idx])
                            self.pgltd3std[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF0D3_"):
                            self.pglf0d3[glnum-1] = float(e[value_idx])
                            self.pglf0d3std[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF0(instant)_"):
                            self.pglf0ins[glnum-1] = float(e[value_idx])
                            self.pglf0insstd[glnum-1] = float(e[3])
                        elif e[0].startswith("GLF0(T="):
                            self.pglf0tg[glnum-1] = float(e[value_idx])
                            self.pglf0tgstd[glnum-1] = float(e[3])
                            results["Tau_g"] = int(e[0].split("=")[1].split(")")[0])
                            glkey = "GLF0(T=tau_g)"
                        if glnum > 1:
                            glep1 = std_error(self.pglep[glnum-1], self.pglepstd[glnum-1])
                            glep2 = std_error(self.pglep[glnum-2], self.pglepstd[glnum-2])
                            delta_glep = glep1 - glep2
                            self.waittime[glnum-1] = delta_glep.n
                            results["Waiting time"] = delta_glep.n
                            results["Waiting time std"] = delta_glep.s
                        glkeystd = glkey + " std"
                        results[glkey] = float(e[value_idx])
                        results[glkeystd] = float(e[3])
                else:
                    if e[0].startswith("TN"):
                        if e[0].startswith("TNRedAmp"):
                            self.redamp = float(e[value_idx])
                            self.redampstd = float(e[3])
                        elif e[0].startswith("TNRedGam"):
                            self.redgam = float(e[value_idx])
                            self.redgamstd = float(e[3])
                        elif e[0].startswith("TNEF"):
                            jbfb = e[0].split()[2]
                            if jbfb == "jbafb":
                                self.ef_jbafb = float(e[value_idx])
                                self.ef_jbafbstd = float(e[3])
                            elif jbfb == "jbdfb":
                                self.ef_jbdfb = float(e[value_idx])
                                self.ef_jbdfbstd = float(e[3])
                        elif e[0].startswith("TNEQ"):
                            jbfb = e[0].split()[2]
                            if jbfb == "jbafb":
                                self.eq_jbafb = float(e[value_idx])
                                self.eq_jbafbstd = float(e[3])
                            elif jbfb == "jbdfb":
                                self.eq_jbdfb = float(e[value_idx])
                                self.eq_jbdfbstd = float(e[3])
                        results[e[0]] = float(e[value_idx])
                        results[e[0]+" std"] = float(e[3])
        return results

    def read_final_par(self, largeglitch=[1]):
        '''Read rough estimates of glitch parameters by tempo2 for small glitches from final par file'''
        for glitchnum in range(self.max_glitch):
            if glitchnum+1 not in largeglitch:
                self.gli_slt[str(glitchnum+1)] = {}
                glf0 = std_error(self.pglf0[glitchnum], self.pglf0std[glitchnum])
                glf1 = std_error(self.pglf1[glitchnum], self.pglf1std[glitchnum])
                glf2 = std_error(self.pglf2[glitchnum], self.pglf2std[glitchnum])
                glf0ins = glf0
                glf0tg = glf1 * self.taug[glitchnum] + 0.5 * glf2 * self.taug[glitchnum]**2
                if self.pglf0d[glitchnum] != 0:
                    glf0d = std_error(self.pglf0d[glitchnum], self.pglf0dstd[glitchnum])
                    gltd = std_error(self.pgltd[glitchnum], self.pgltdstd[glitchnum])
                    glf0ins += glf0d
                    glf0tg[glitchnum] += glf0d*(np.exp(-self.taug[glitchnum]/gltd)-1)
                if self.pglf0d2[glitchnum] != 0:
                    glf0d2 = std_error(self.pglf0d2[glitchnum], self.pglf0d2std[glitchnum])
                    gltd2 = std_error(self.pgltd2[glitchnum], self.pgltd2std[glitchnum])
                    glf0ins += glf0d2
                    glf0tg[glitchnum] += glf0d2*(np.exp(-self.taug[glitchnum]/gltd2)-1)
                if self.pglf0d3[glitchnum] != 0:
                    glf0d3 = std_error(self.pglf0d3[glitchnum], self.pglf0d3std[glitchnum])
                    gltd3 = std_error(self.pgltd3[glitchnum], self.pgltd3std[glitchnum])
                    glf0ins += glf0d3
                    glf0tg[glitchnum] += glf0d3*(np.exp(-self.taug[glitchnum]/gltd3)-1)
                self.gli_slt[str(glitchnum+1)]["GLF0(instant)"] = glf0ins.n
                self.gli_slt[str(glitchnum+1)]["GLF0(instant) std"] = glf0ins.s
                self.gli_slt[str(glitchnum+1)]["GLF0(T=tau_g)"] = glf0tg.n
                self.gli_slt[str(glitchnum+1)]["GLF0(T=tau_g) std"] = glf0tg.s
                self.gli_slt[str(glitchnum+1)]["Tau_g"] = self.taug[glitchnum]
                if glitchnum > 0:
                    glep1 = std_error(self.pglep[glitchnum], self.pglepstd[glitchnum])
                    glep2 = std_error(self.pglep[glitchnum-1], self.pglepstd[glitchnum-1])
                    delta_glep = glep1 - glep2
                    self.waittime[glitchnum] = delta_glep.n
                    self.gli_slt[str(glitchnum+1)]["Waiting time"] = delta_glep.n
                    self.gli_slt[str(glitchnum+1)]["Waiting time std"] = delta_glep.s
        with open(self.par) as f:
            f.readline()
            for line in f:
                e=line.strip().split()
                if e[0].startswith("GL"):
                    glnum = int(e[0].split("_")[-1])
                    if glnum not in largeglitch:
                        glkey = e[0].split("_")[0]
                        std = np.nan
                        if len(e) > 3:
                            std = float(e[3])
                        if e[0].startswith("GLF0(T="):
                            self.gli_slt[str(glnum)]["Tau_g"] = int(e[0].split("=")[1].split(")")[0])
                            glkey = "GLF0(T=tau_g)"
                        glkeystd = glkey + " std"
                        self.gli_slt[str(glnum)][glkey] = float(e[1])
                        self.gli_slt[str(glnum)][glkeystd] = std
        for glitchnum in range(self.max_glitch):
            if glitchnum+1 not in largeglitch:
                print("<<< The solution for glitch {} is {} >>>".format(glitchnum+1, self.gli_slt[str(glitchnum+1)]))

    def extract_results(self, finalpar=None):
        '''Extract useful results from final par'''
        if finalpar is None:
            finalpar = self.par
        spin, position, toa, red, white, glitch, recovery, summary = self.load_par(finalpar, glf0t=self.taug)
        self.psrn, self.F0, self.F1, self.F2, self.F0std, self.F1std, self.F2std = spin
        self.ra, self.dec, self.pmra, self.pmdec, self.PX, self.DM = position
        self.start, self.finish, self.pepoch = toa
        #self.redamp, self.redgam, self.redc = red
        #self.ef_jbafb, self.ef_jbdfb, self.eq_jbafb, self.eq_jbdfb = white
        #self.pglep, self.pglf0, self.pglf1, self.pglf2, self.pglf0ins, self.pglf0tg = glitch
        #self.pglf0d, self.pgltd, self.pglf0d2, self.pgltd2, self.pglf0d3, self.pgltd3 = recovery
        #self.max_glitch, self.waittime, self.taug, self.numofexp = summary
        self.P, self.P_dot, self.P_ddot = self.calculate_period()
        self.E_dot, self.tau_c, self.B_sur, self.brake = self.calculate_derived()
        self.delete_null()
        self.print_info()
        self.spin_slt["Pulsar Name"] = self.psrn
        self.spin_slt["F0"] = self.F0
        self.spin_slt["F0 std"] = self.F0std
        self.spin_slt["F1"] = self.F1
        self.spin_slt["F1 std"] = self.F1std
        self.spin_slt["F2"] = self.F2
        self.spin_slt["F2 std"] = self.F2std
        self.spin_slt["P"] = self.P.n
        self.spin_slt["P std"] = self.P.s
        self.spin_slt["P_dot"] = self.P_dot.n
        self.spin_slt["P_dot std"] = self.P_dot.s
        self.spin_slt["P_ddot"] = self.P_ddot.n
        self.spin_slt["P_ddot std"] = self.P_ddot.s
        self.spin_slt["Total glitches"] = self.max_glitch
        self.spin_slt["E_dot"] = self.E_dot
        self.spin_slt["Tau_c"] = self.tau_c
        self.spin_slt["B_sur"] = self.B_sur
        self.spin_slt["Braking index"] = self.brake
        fnlrslt = "../slt_"+self.psrn+".csv"
        entry = []
        spin_idx = ["F0", "F0 std", "F1", "F1 std", "F2", "F2 std"]
        period_idx = ["P", "P std", "P_dot", "P_dot std", "P_ddot", "P_ddot std"]
        derived_idx = ["E_dot", "Tau_c", "B_sur", "Braking index"]
        glitch_idx = ["GLEP", "GLEP std", "GLF0", "GLF0 std", "GLF1", "GLF1 std", "GLF2", "GLF2 std"]
        alter_idx = ["GLF0(instant)", "GLF0(instant) std", "GLF0(T=tau_g)", "GLF0(T=tau_g) std", "Tau_g", "Waiting time"]
        recovery_idx = ["GLF0D", "GLF0D std", "GLTD", "GLTD std", "GLF0D2", "GLF0D2 std", "GLTD2", "GLTD2 std", "GLF0D3", "GLF0D3 std", "GLTD3", "GLTD3 std"]
        red_idx = ["TNRedAmp", "TNRedAmp std", "TNRedGam", "TNRedGam std"]
        white_idx = ["TNEF -be jbafb", "TNEF -be jbafb std", "TNEF -be jbdfb", "TNEF -be jbdfb std", "TNEQ -be jbafb", "TNEQ -be jbafb std", "TNEQ -be jbdfb", "TNEQ -be jbdfb std"]
        sum_idx = spin_idx + period_idx + derived_idx + glitch_idx + alter_idx + recovery_idx + red_idx #+ white_idx
        #unit = np.full(len(sum_idx), None)
        #unit[12:15] = ["erg/s", "kyr", "1e12G"]
        #col_id = pd.MultiIndex.from_tuples([(i, j) for i, j in zip(sum_idx, unit)], names=["Parameters", "Units"])
        # To do: add units as second column index
        for glitchnum in range(self.max_glitch):
            sum_slt = {**self.spin_slt, **self.gli_slt[str(glitchnum+1)], **self.noise_slt}
            series_slt = pd.Series(sum_slt, index=sum_idx)
            entry.append(series_slt)
        fr_name = ["Pulsar name", "Glitch No."]
        fr_id = pd.MultiIndex.from_tuples([(self.psrn, i+1) for i in range(self.max_glitch)], names=fr_name)
        frame_slt = pd.DataFrame(entry, index=fr_id)
        frame_slt.to_csv(fnlrslt)

    def calculate_period(self):
        '''Calculate period and period derivatives of pulsars from spin parameters'''
        F0 = std_error(self.F0, self.F0std)
        F1 = std_error(self.F1, self.F1std)
        F2 = std_error(self.F2, self.F2std)
        P0 = 1/F0
        P1 = -F1/(F0**2)
        P2 = 2*(F1**2)/(F0**3) - F2/(F0**2)
        return P0, P1, P2

    def calculate_derived(self):
        '''Calculate derived quantities of pulsars from spin parameters'''
        # Assume canonical mass M_ns = 1.4*M_sun, r_ns = 10km, i.e., moment of inertia I = 10^45 g*cm^2 = 10^38 kg*m^2
        #E_dot = -4e38*(np.pi**2)*self.F0*self.F1 # in W
        E_dot = -4e45*(np.pi**2)*self.F0*self.F1 # in erg/s
        #tau_c = -0.5*self.F0/self.F1 # in seconds
        tau_c = -0.5/(86400*365.2425e3)*self.F0/self.F1 # in kyr
        B_surface = 3.2e7*np.sqrt(-self.F1/self.F0**3) # in T(10^12)Gauss
        braking_index = self.F0*self.F2/(self.F1**2)
        return E_dot, tau_c, B_surface, braking_index

    def noglitch_par(self):
        ''' Make a copy of par file without glitch parameters'''
        parlines = []
        print("### Final par for test par is:", self.par)
        with open(self.par) as f:
            for line in f:
                if line.startswith("GLEP_"):
                    parlines.append(line)
                elif line.startswith("GL") or line.startswith("TNRed"):
                    continue
                elif line.startswith("TNEF") or line.startswith("TNEQ"):
                    parlines.append(line)
                elif any(line.startswith(ls) for ls in ["RA", "DEC", "JUMP", "DM ", "PM", "PX", "F0", "F1", "F2"]):
                    parlines.append(line)
                else: # Turn off tempo2 fitting of all other parameters
                    e = line.split()
                    if len(e) > 3 and float(e[2]) == 1:
                        newline = e[0]+"   "+e[1]+"   "+e[3]+"\n"
                        parlines.append(newline)
                    elif len(e) == 3 and float(e[2]) == 1:
                        newline = e[0]+"   "+e[1]+"\n"
                        parlines.append(newline)
                    else:
                        parlines.append(line)
        self.testpar = "tst_"+self.par.split("_", 1)[1]
        with open(self.testpar, "w") as newf:
            newf.writelines(parlines)

    def sf_create_global(self, leading, trailing, width, step):
        ''' Set fit timespan using a global par file'''
        try:
            os.remove("global.par")
        except OSError:
            pass
        with open("global.par", "a") as glob:
            for ge in self.pglep:
                if ge is not None:
                    if leading < ge and trailing >= ge:
                        old_trail = trailing
                        trailing = ge - 0.01*step
                        if trailing - leading < width / 2.0:
                            leading = ge + 0.01*step
                            trailing = old_trail
            glob.write("START {} 1\n".format(leading))
            glob.write("FINISH {} 1".format(trailing))
        return leading, trailing

    def sf_run_fit(self, epoch, F0e=1e-7, F1e=1e-14, F2e=1e-17):
        '''Run tempo2 and fit for parameters'''
        epoch = str(epoch)
        command = ["tempo2", "-f", self.testpar, self.tim, "-nofit", "-global", "global.par",
                   "-fit", "F0", "-fit", "F1", "-fit", "F2", "-epoch", epoch]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
        while proc.poll() is None:
            line = proc.stdout.readline().decode("utf-8")
            fields = line.split()
            if len(fields) > 4:
                if fields[0] == "PEPOCH":
                    pepoch = fields[3]
                if fields[0] == "F0":
                    F0 = fields[3]
                    F0_e = fields[4]
                    if not 0 < abs(float(F0_e)) < F0e:
                        return None
                if fields[0] == "F1":
                    F1 = fields[3]
                    F1_e = fields[4]
                    if not 0 < abs(float(F1_e)) < F1e:
                        return None
                if fields[0] == "F2":
                    F2 = fields[3]
                    F2_e = fields[4]
                    if not 0 < abs(float(F2_e)) < F2e:
                        return None
        try:
            return pepoch, F0, F0_e, F1, F1_e, F2, F2_e
        except UnboundLocalError:
            return None

    def sf_main(self, width, step, F0e=1e-7, F1e=1e-14, F2e=1e-17):
        '''Main function for stride fitting'''
        if step <= 0:
            step = 3*int(self.cadence)
        if width < 2*step:
            width = 2*step
        print("Stride fitting with box width %d and step size %d"%(width, step))
        first, last = self.minmjds, self.maxmjds
        leading = first
        trailing = first + width
        counter = 0
        self.epochfile = self.psrn+"_g"+str(self.max_glitch)+"_w"+str(int(width))+"_s"+str(int(step))+"_epoch.txt"
        with open(self.epochfile, "w") as f1:
            while trailing <= last:
                leading, trailing = self.sf_create_global(leading, trailing, width, step)
                epoch = leading + ((trailing - leading)/2.0)
                print(leading, trailing, epoch, file=f1)
                counter += 1
                leading = first + counter*step
                trailing = first + width + counter*step
        starts, ends, fitepochs = np.loadtxt(self.epochfile, unpack=True)
        self.datafile = self.psrn+"_g"+str(self.max_glitch)+"_w"+str(int(width))+"_s"+str(int(step))+"_data.txt"
        with open(self.datafile, "w") as f2:
            for i, (start_value, end_value) in enumerate(zip(starts, ends)):
                self.sf_create_global(start_value, end_value, width, step)
                epoch = fitepochs[i]
                out = self.sf_run_fit(epoch, F0e, F1e, F2e)
                os.remove("global.par")
                if out:
                    print(out[0], out[1], out[2], out[3], out[4], out[5], out[6], starts[i], ends[i], file=f2)

    def sf_calculate_data(self, save=True, plot=False):
        '''Load stride fitting results, plot stride fitting results and save to text files'''
        # sft, f0, f0e, f1, f1e, f2, f2e, mjds, mjdf = np.loadtxt(self.datafile, unpack=True)
        sft, f0, f0e, f1, f1e = np.loadtxt(self.datafile, usecols=(0, 1, 2, 3, 4), unpack=True)
        sfx = mjd2sec(sft, self.pepoch)
        sff1, sff2, sfdf2 = self.psr_taylor_terms(sfx)
        sfglf0, sfglf1, sfglf2, sfexp1, sfexp2, sfexp3, sfdglf1, sfdglf2, sfdexp = self.glitch_terms(sft)
        p1y = (f0 - self.F0 - sff1) #- sff2)
        p2y = (f0 - self.F0 - sff1 - sff2 - sfglf0 - sfglf1 - sfglf2)
        p3y = (f0 - self.F0 - sff1 - sff2 - sfglf0 - sfglf1 - sfglf2 - sfexp1 - sfexp2 - sfexp3)
        p4y = (f1 - self.F1) #- sfdf2)
        p5y = (f1 - self.F1 - sfdf2 - sfdglf1 - sfdglf2 + sfdexp)
        if save:
            with open("panel1_{}.txt".format(self.psrn), "w") as file1:
                for i, value in enumerate(sft):
                    file1.write("%f   %e   %e   \n" % (value, 1e6*(p1y[i]), 1e6*f0e[i]))
                file1.close()
            with open("panel2_{}.txt".format(self.psrn), "w") as file2:
                for i, value in enumerate(sft):
                    file2.write("%f   %e   %e   \n" % (value, 1e6*(p2y[i]), 1e6*f0e[i]))
                file2.close()
            with open("panel3_{}.txt".format(self.psrn), "w") as file3:
                for i, value in enumerate(sft):
                    file3.write("%f   %e   %e   \n" % (value, 1e6*(p3y[i]), 1e6*f0e[i]))
                file3.close()
            with open("panel4_{}.txt".format(self.psrn), "w") as file4:
                for i, value in enumerate(sft):
                    file4.write("%f   %e   %e   \n" % (value, 1e10*(p4y[i]), 1e10*f1e[i]))
                file4.close()
            with open("panel5_{}.txt".format(self.psrn), "w") as file5:
                for i, value in enumerate(sft):
                    file5.write("%f   %e   %e   \n" % (value, 1e10*(p5y[i]), 1e10*f1e[i]))
                file5.close()
        if plot:
            plt.errorbar(sft, 1e6*p1y, yerr=1e6*f0e, marker=".", color="k", ecolor="k", linestyle="None")
            plt.show()
            plt.errorbar(sft, 1e6*p2y, yerr=1e6*f0e, marker=".", color="k", ecolor="k", linestyle="None")
            plt.show()
            plt.errorbar(sft, 1e6*p3y, yerr=1e6*f0e, marker=".", color="k", ecolor="k", linestyle="None")
            plt.show()
            plt.errorbar(sft, 1e10*p4y, yerr=1e10*f1e, marker=".", color="k", ecolor="k", linestyle="None")
            plt.show()
            plt.errorbar(sft, 1e10*p5y, yerr=1e10*f1e, marker=".", color="k", ecolor="k", linestyle="None")
            plt.show()

    def print_info(self, index=None):
        '''Print basic info of pulsar'''
        print("")
        print("<<< Spin and TOA information of pulsar >>>")
        print("Pulsar name:", self.psrn)
        print("F0:", std_error(self.F0, self.F0std))
        print("F1:", std_error(self.F1, self.F1std))
        print("F2:", std_error(self.F2, self.F2std))
        print("Period epoch:", self.pepoch)
        print("TOA start:", self.minmjds)
        print("TOA finish:", self.maxmjds)
        print("TOA length:", self.toanum)
        print("Cadence", self.cadence)
        print("")
        print("<<< Noise information of pulsar >>>")
        print("Red noise amplitude:", self.redamp)
        print("Red noise gamma:", self.redgam)
        print("Red noise constant:", self.redc)
        print("White noise TNEF jbafb:", self.ef_jbafb)
        print("White noise TNEF jbdfb:", self.ef_jbdfb)
        print("White noise TNEQ jbafb:", self.ef_jbafb)
        print("White noise TNEQ jbdfb:", self.ef_jbdfb)
        print("")
        print("<<< Derived information of pulsar >>>")
        print("Spin-down luminosity E_dot: {} erg/s".format(self.E_dot))
        print("Characteristic age tau_c: {} kyr".format(self.tau_c))
        print("Surface magnetic field strength B_sur: {} TG".format(self.B_sur))
        print("Braking index n: {}".format(self.brake))
        print("")
        self.print_glitch_info(index)

    def print_glitch_info(self, index=None):
        '''Print basic info of the index glitch in the pulsar'''
        if isinstance(index, int) and 0 < index <= self.max_glitch:
            print("<<< Glitch information of glitch No.{} >>>".format(index))
            print("Glitch epoch GLEP_{}: {}".format(index, self.pglep[index-1]))
            print("GLF0_{}: {}".format(index, self.pglf0[index-1]))
            print("GLF1_{}: {}".format(index, self.pglf1[index-1]))
            print("GLF2_{}: {}".format(index, self.pglf2[index-1]))
            if self.pglf0d[index-1] != 0:
                print("GLF0D_{}: {}".format(index, self.pglf0d[index-1]), " - GLTD_{}: {}".format(index, self.pgltd[index-1]))
            if self.pglf0d2[index-1] != 0:
                print("GLF0D2_{}: {}".format(index, self.pglf0d2[index-1]), " - GLTD2_{}: {}".format(index, self.pgltd2[index-1]))
            if self.pglf0d3[index-1] != 0:
                print("GLF0D3_{}: {}".format(index, self.pglf0d3[index-1]), " - GLTD3_{}: {}".format(index, self.pgltd3[index-1]))
            print("Initial jump (GLFO(instant)_{}): {}".format(index, self.pglf0ins[index-1]))
            print("Decay jump (GLFO(T=tau_g)_{}): {}".format(index, self.pglf0tg[index-1]))
            print("Waiting time (Inactive time before glitch No.{}): {}".format(index, self.waittime[index-1]))
            print("tau_g:", self.taug[index-1])
            print("Number of exponentials:", int(self.numofexp[index-1]))
            print("")
        else:
            for i in range(self.max_glitch):
                self.print_glitch_info(i+1)

    def glitch_terms(self, t, gn=None):
        '''Calculate the glitch terms for MJD arrays t and the No.gn glitch in pulsar: x = time since glitch epoch in seconds'''
        glf0 = np.zeros_like(t)
        glf1 = np.zeros_like(t)
        glf2 = np.zeros_like(t)
        exp1 = np.zeros_like(t)
        exp2 = np.zeros_like(t)
        exp3 = np.zeros_like(t)
        dglf1 = np.zeros_like(t)
        dglf2 = np.zeros_like(t)
        dexp = np.zeros_like(t)
        if isinstance(gn, int) and 0 <= gn < self.max_glitch:
            glep = self.pglep[gn]
            x = mjd2sec(t, glep)
            e1 = np.zeros_like(t)
            e2 = np.zeros_like(t)
            e3 = np.zeros_like(t)
            glf0[x > 0] += self.pglf0[gn]
            glf1[x > 0] += self.pglf1[gn] * x[x > 0]
            glf2[x > 0] += 0.5 * self.pglf2[gn] * x[x > 0]**2
            e1[x > 0] += self.pglf0d[gn] * np.exp(-x[x > 0] / (self.pgltd[gn]*86400.0))
            e2[x > 0] += self.pglf0d2[gn] * np.exp(-x[x > 0] / (self.pgltd2[gn]*86400.0))
            e3[x > 0] += self.pglf0d3[gn] * np.exp(-x[x > 0] / (self.pgltd3[gn]*86400.0))
            dglf1[x > 0] += self.pglf1[gn]
            dglf2[x > 0] += self.pglf2[gn] * x[x > 0]
            dexp[x > 0] += e1[x > 0] / (self.pgltd[gn]*86400) + e2[x > 0] / (self.pgltd2[gn]*86400) + e3[x > 0] / (self.pgltd3[gn]*86400)
            exp1 += e1
            exp2 += e2
            exp3 += e3
        else:
            for i in range(self.max_glitch):
                glep = self.pglep[i]
                x = mjd2sec(t, glep)
                e1 = np.zeros_like(t)
                e2 = np.zeros_like(t)
                e3 = np.zeros_like(t)
                glf0[x > 0] += self.pglf0[i]
                glf1[x > 0] += self.pglf1[i] * x[x > 0]
                glf2[x > 0] += 0.5 * self.pglf2[i] * x[x > 0]**2
                e1[x > 0] += self.pglf0d[i] * np.exp(-x[x > 0] / (self.pgltd[i]*86400.0))
                e2[x > 0] += self.pglf0d2[i] * np.exp(-x[x > 0] / (self.pgltd2[i]*86400.0))
                e3[x > 0] += self.pglf0d3[i] * np.exp(-x[x > 0] / (self.pgltd3[i]*86400.0))
                dglf1[x > 0] += self.pglf1[i]
                dglf2[x > 0] += self.pglf2[i] * x[x > 0]
                dexp[x > 0] += e1[x > 0] / (self.pgltd[i]*86400) + e2[x > 0] / (self.pgltd2[i]*86400) + e3[x > 0] / (self.pgltd3[i]*86400)
                exp1 += e1
                exp2 += e2
                exp3 += e3
        return glf0, glf1, glf2, exp1, exp2, exp3, dglf1, dglf2, dexp

    def psr_taylor_terms(self, x):
        '''Calculate the pulsar taylor series terms for array x in second:
             x = time since period epoch in seconds'''
        tf1 = self.F1 * x
        tf2 = 0.5 * self.F2 * x * x
        tdf2 = self.F2 * x
        return tf1, tf2, tdf2

    def mask_glep(self, t, array):
        '''Mask data at GLEPs for MJD arrays t'''
        mask_index = []
        # for i, value in enumerate(t):
        for i in range(len(t)):   # using enumerate
            # for gi in range(self.max_glitch):
                # if t[i] <= self.pglep[gi] < t[i+1]:
            if any(t[i] <= glep < t[i+1] for glep in self.pglep):
                mask_index.append(i)   # or i+1
        mc = ma.array(array)
        mc[mask_index] = ma.masked
        return mc

    def pp_create_files(self):
        '''Call tempo2 to generate files for pulsar plots'''
        print("### Final par for stride fitting is:", self.par)
        print("### Test par for stride fitting is:", self.testpar)
        subprocess.call(["tempo2", "-output", "exportres", "-f", self.testpar, self.tim, "-nofit"])
        os.rename("out.res", "out2_{}.res".format(self.psrn))
        subprocess.call(["tempo2", "-output", "exportres", "-f", self.par, self.tim, "-writeres"])
        os.rename("param.labels", "param_{}.labels".format(self.psrn))
        os.rename("param.vals", "param_{}.vals".format(self.psrn))
        os.rename("cov.matrix", "cov_{}.matrix".format(self.psrn))
        os.rename("tnred.meta", "tnred_{}.meta".format(self.psrn))
        os.rename("out.res", "out_{}.res".format(self.psrn))
        os.rename("prefit.res", "prefit_{}.res".format(self.psrn))
        os.rename("postfit.res", "postfit_{}.res".format(self.psrn))
        os.rename("awhite.res", "awhite_{}.res".format(self.psrn))
        os.rename("design.matrix", "design_{}.matrix".format(self.psrn))
        os.rename("constraints.matrix", "constraints_{}.matrix".format(self.psrn))
        os.rename("adesign.matrix", "adesign_{}.matrix".format(self.psrn))

    def pp_calculate_data(self, start=None, finish=None):
        lab = np.loadtxt("param_{}.labels".format(self.psrn), dtype=np.str).T
        beta = np.loadtxt("param_{}.vals".format(self.psrn))
        meta = np.loadtxt("tnred_{}.meta".format(self.psrn), usecols=(1))
        omega, epoch = meta[0], meta[1]
        rx, ry, re = np.loadtxt("out_{}.res".format(self.psrn), usecols=(0, 5, 6), unpack=True)
        if isinstance(start, int) and isinstance(finish, int):
            t = np.linspace(start-0.5, finish+0.5, 1000)
        else:
            t = np.linspace(self.start-0.5, self.finish+0.5, 1000)
        y = np.zeros_like(t)
        cosidx=lab[1]=="param_red_cos"
        sinidx=lab[1]=="param_red_sin"
        maxwav=400
        nc=ns=0
        for i, (vcos, vsin) in enumerate(zip(cosidx, sinidx)):
            if vcos:
                nc+=1
                if nc > maxwav:
                    cosidx[i:] = False
            if vsin:
                ns+=1
                if ns > maxwav:
                    sinidx[i:] = False
            if nc>maxwav and ns>maxwav:
                break
        nwav = np.sum(sinidx)
        beta_mod = beta[np.logical_or(sinidx,cosidx)]
        M = np.zeros((2*nwav,len(t)))
        M2 = np.zeros((2*nwav,len(rx)))
        dM = np.zeros_like(M)
        ddM = np.zeros_like(M)
        with open("white_{}.asc".format(self.psrn),"w") as f:
            f.write("WAVE_OM {}\n".format(omega))
            f.write("WAVEEPOCH {}\n".format(epoch))
            for i in range(min(256,nwav)):
                f.write("WAVE{}  {}  {}\n".format(i+1, -beta_mod[i], -beta_mod[i+nwav]))
        #print("set up matricies")
        freqs=[]
        pwrs=np.power(beta_mod[:nwav],2) + np.power(beta_mod[nwav:],2)
        for i in range(nwav):
            omegai = omega*(i+1.0)
            M[i]        = np.sin(omegai * (t-epoch))
            M[i+nwav]   = np.cos(omegai * (t-epoch))
            freqs.append(365.25*omegai/2.0/np.pi)
            dM[i]      = -self.F0*omegai*M[i+nwav]/86400.0
            dM[i+nwav] = self.F0*omegai*M[i]/86400.0
            ddM[i]      = 1e15*self.F0*omegai*omegai*M[i]/(86400.0**2)
            ddM[i+nwav] = 1e15*self.F0*omegai*omegai*M[i+nwav]/(86400.0**2)                 
            M2[i]       = np.sin(omegai * (rx-epoch))
            M2[i+nwav]  = np.cos(omegai * (rx-epoch)) 
        #print("Do linear algebra")
        freqs=np.array(freqs)
        maxP=2*np.pi/omegai
        tt = mjd2sec(t, self.pepoch)
        M = M.T
        dM = dM.T
        ddM = ddM.T
        M2 = M2.T
        y = M.dot(beta_mod)
        yf = dM.dot(beta_mod)
        yd = ddM.dot(beta_mod)
        y_dat = M2.dot(beta_mod)
        yf2 = yf + (0.5*self.F2*tt*tt) #yf2 = yf + (0.5*F2*tt*tt + tt*F1 + F0)
        yd_model = np.zeros_like(yd)
        yd_model += 1e15*(self.F2*tt + self.F1)
        for i, ge in enumerate(self.pglep):
            gt = mjd2sec(t, ge)
            yf2[t>ge] +=  self.pglf0[i] + self.pglf1[i] * gt[t>ge] + 0.5 * self.pglf2[i] * gt[t>ge]**2 #bug
            yd_model[t>ge] += 1e15 * (self.pglf1[i] + self.pglf2[i] * gt[t>ge]) #bug
            if self.pglf0d[i] != 0:
                yf2[t>ge] += self.pglf0d[i] * np.exp(-(t[t>ge]-self.pglep[i])/self.pgltd[i])
                yf2[t>ge] += self.pglf0d2[i] * np.exp(-(t[t>ge]-self.pglep[i])/self.pgltd2[i])
                yf2[t>ge] += self.pglf0d3[i] * np.exp(-(t[t>ge]-self.pglep[i])/self.pgltd3[i])
                yd_model[t>ge] -= 1e15*self.pglf0d[i] * np.exp(-(t[t>ge]-self.pglep[i])/self.pgltd[i]) / (self.pgltd[i]*86400.0)
                yd_model[t>ge] -= 1e15*self.pglf0d2[i] * np.exp(-(t[t>ge]-self.pglep[i])/self.pgltd2[i]) / (self.pgltd2[i]*86400.0)
                yd_model[t>ge] -= 1e15*self.pglf0d3[i] * np.exp(-(t[t>ge]-self.pglep[i])/self.pgltd3[i]) / (self.pgltd3[i]*86400.0)
        yd2 = yd + yd_model
        ry += np.mean(y_dat-ry)
        with open("ifunc_{}.asc".format(self.psrn), "w") as f:
            for i, value in enumerate(t):
                f.write("{}  {}  {}\n".format(i+1, value, y[i])) #f.write("IFUNC{}  {}  {} {}\n".format(i+1, t[i], -y[i], 0))
        with open("deltanu_{}.asc".format(self.psrn), "w") as f:
            for i, value in enumerate(yf2):
                f.write("{} {}\n".format(t[i], value))
        with open("nudot_{}.asc".format(self.psrn), "w") as f:
            for i, value in enumerate(yd):
                f.write("{} {} {} {}\n".format(t[i], value, yd_model[i], yd2[i]))
        return rx, ry, re, y_dat, freqs, pwrs

    # def measure_prior
    # def set_prior


class Glitch(Pulsar):
    '''A Glitch subclass of Pulsar class consists glitch parameters and glitch model info'''
    def __init__(self, Pulsar, index):
        '''Initialize pulsar class with corresponding parameter file and TOA file'''
        self.parentpsr = Pulsar
        super().__init__(Pulsar.par, Pulsar.tim)
        if not (isinstance(index, int) and 0 < index <= Pulsar.max_glitch):
            index = Pulsar.max_glitch+1
            Pulsar.max_glitch += 1
        self.index = index-1
        self.inherit_pulsar()
        # self.create_new(self)

    def inherit_pulsar(self):
        '''Inherit glitch parameters from Pulsar info'''
        self.pglep = self.pglep[self.index]
        self.pglf0 = self.pglf0[self.index]
        self.pglf1 = self.pglf1[self.index]
        self.pglf2 = self.pglf2[self.index]
        self.pglf0d = self.pglf0d[self.index]
        self.pgltd = self.pgltd[self.index]
        self.pglf0d2 = self.pglf0d2[self.index]
        self.pgltd2 = self.pgltd2[self.index]
        self.pglf0d3 = self.pglf0d3[self.index]
        self.pgltd3 = self.pgltd3[self.index]
        self.taug = self.taug[self.index]
        self.pglf0ins = self.pglf0ins[self.index]
        self.pglf0tg = self.pglf0tg[self.index]
