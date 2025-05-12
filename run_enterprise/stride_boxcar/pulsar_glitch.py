#!/usr/bin/env python
'''Pulsar glitch module for processing glitch data.
    Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).'''

from __future__ import print_function
# import argparse
import sys
import os
import subprocess
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.ticker import FormatStrFormatter
# from scipy.optimize import curve_fit
# from astropy import coordinates as coord
# from astropy import units as u
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid.inset_locator import inset_axes



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

class Pulsar:
    '''A pulsar class consists basic info of pulsar'''
    def __init__(self, parfile, timfile, tidy=False, glf0t="GLTD"):
        '''Initialize pulsar class with corresponding par file and tim file'''
        self.par = parfile
        self.tim = timfile
        self.psrn = None
        self.ra = None
        self.dec = None
        self.PB = None
        self.F0 = 0
        self.F1 = 0
        self.F2 = 0
        self.start = None
        self.finish = None
        self.pepoch = None
        self.tidypar = None
        self.testpar = None
        self.truth = None
        self.epochfile = None
        self.datafile = None
        self.load_info(tidy=tidy, glf0t=glf0t)

    def load_info(self, tidy=False, glf0t="GLTD"):
        '''Load basic info of pulsar class from par file and tim file'''
        if tidy==False or self.tidypar==None:
            loadpar = self.par
        else:
            loadpar = self.tidypar
        self.max_glitch = 0
        self.pglep = np.zeros(100)
        self.pglf0 = np.zeros(100)
        self.pglf1 = np.zeros(100)
        self.pglf2 = np.zeros(100)
        self.pglf0d = np.zeros(100)
        self.pgltd = np.ones(100)
        self.pglf0d2 = np.zeros(100)
        self.pgltd2 = np.ones(100)
        self.pglf0d3 = np.zeros(100)
        self.pgltd3 = np.ones(100)
        self.taug = 200*np.ones(100)
        self.pglf0ins = np.zeros(100)
        self.pglf0tg = np.zeros(100)
        self.numofexp = np.zeros(100)
        self.redamp = None
        self.redgam = None
        with open(loadpar) as f1:
            for line in f1:
                line = line.strip()
                e = line.split()
                if e[0] == "PSRJ":
                    self.psrn = e[1]
                if e[0] == "RAJ":
                    self.ra = e[1]
                if e[0] == "DECJ":
                    self.dec = e[1]
                if e[0].startswith("GLEP_"):
                    i = int(e[0][5:])
                    self.pglep[i-1] = float(e[1])
                    self.max_glitch = max(i, self.max_glitch)
                if e[0].startswith("GLF0_"):
                    i = int(e[0][5:])
                    self.pglf0[i-1] = float(e[1])
                if e[0].startswith("GLF1_"):
                    i = int(e[0][5:])
                    self.pglf1[i-1] = float(e[1])
                if e[0].startswith("GLF2_"):
                    i = int(e[0][5:])
                    self.pglf2[i-1] = float(e[1])
                if e[0].startswith("GLTD_"):
                    i = int(e[0][5:])
                    self.pgltd[i-1] = float(e[1])
                if e[0].startswith("GLF0D_"):
                    i = int(e[0][6:])
                    self.pglf0d[i-1] = float(e[1])
                    self.numofexp[i-1] = max(1, self.numofexp[i-1])
                if e[0].startswith("GLTD2_"):
                    i = int(e[0][6:])
                    self.pgltd2[i-1] = float(e[1])
                if e[0].startswith("GLF0D2_"):
                    i = int(e[0][7:])
                    self.pglf0d2[i-1] = float(e[1])
                    self.numofexp[i-1] = max(2, self.numofexp[i-1])
                if e[0].startswith("GLTD3_"):
                    i = int(e[0][6:])
                    self.pgltd3[i-1] = float(e[1])
                if e[0].startswith("GLF0D3_"):
                    i = int(e[0][7:])
                    self.pglf0d3[i-1] = float(e[1])
                    self.numofexp[i-1] = max(3, self.numofexp[i-1])
                if e[0].startswith("GLF0(T="):
                    i = int(e[0].split('_')[-1])
                    numb = e[0].split('=')[1]
                    self.taug[i-1] = int(numb.split(')')[0])
                if e[0] == "PB":
                    self.PB = float(e[1])
                if e[0] == "F0":
                    self.F0 = float(e[1])
                if e[0] == "F1":
                    self.F1 = float(e[1])
                if e[0] == "F2":
                    self.F2 = float(e[1])
                if e[0] == "START":
                    self.start = float(e[1])
                if e[0] == "FINISH":
                    self.finish = float(e[1])
                if e[0] == "PEPOCH":
                    self.pepoch = float(e[1])
                if e[0] == "TNRedAmp":
                    self.redamp = float(e[1])
                if e[0] == "TNRedGam":
                    self.redgam = float(e[1])
        for i in range(self.max_glitch):
            if glf0t == "GLTD":
                if self.pgltd[i] != 1:
                    self.taug[i] = int(self.pgltd[i])
                if self.pgltd2[i] != 1:
                    self.taug[i] = int(min(self.taug[i], self.pgltd2[i]))
                if self.pgltd3[i] != 1:
                    self.taug[i] = int(min(self.taug[i], self.pgltd3[i]))
            elif i<len(glf0t):
                self.taug[i] = int(glf0t[i])
            else:
                pass
            self.pglf0ins[i] = self.pglf0[i]
            self.pglf0tg[i] = self.pglf1[i] * self.taug[i] + 0.5 * self.pglf2[i] * self.taug[i]**2
            if self.pglf0d[i] != 0:
                self.pglf0ins[i] += self.pglf0d[i]
                self.pglf0tg[i] += self.pglf0d[i]*(np.exp(-self.taug[i]/self.pgltd[i])-1)
            if self.pglf0d2[i] != 0:
                self.pglf0ins[i] += self.pglf0d2[i]
                self.pglf0tg[i] += self.pglf0d2[i]*(np.exp(-self.taug[i]/self.pgltd2[i])-1)
            if self.pglf0d3[i] != 0:
                self.pglf0ins[i] += self.pglf0d3[i]
                self.pglf0tg[i] += self.pglf0d3[i]*(np.exp(-self.taug[i]/self.pgltd3[i])-1)
        # Load tim file info
        self.minmjds = 1e6
        self.maxmjds = 0
        self.toanum = 0
        self.toaseries = []
        with open(self.tim) as f2:
            for line in f2:
                e = line.split()
                if len(e) > 2.0 and e[0] != "C" and "-pn" in e:
                    self.toaseries.append(float(e[2]))
                    self.minmjds = min(self.minmjds, float(e[2]))
                    self.maxmjds = max(self.maxmjds, float(e[2]))
                    self.toanum += 1
        self.toaseries.sort()
        self.toaseries = np.array(self.toaseries)
        self.toaspan = self.maxmjds-self.minmjds
        self.cadence = self.toaspan/(self.toanum-1)

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
        print('Threshold of gap:', gap)
        maski = x<0
        counter = 1
        for i, value in enumerate(self.toainterval):
            if value >= gap:
                print('The No.%i gap (%f) in TOA is from %f to %f'%(counter, value, self.toaseries[i-1], self.toaseries[i]))
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
        self.taug = self.taug[:self.max_glitch]
        self.pglf0ins = self.pglf0ins[:self.max_glitch]
        self.pglf0tg = self.pglf0tg[:self.max_glitch]
        self.numofexp = self.numofexp[:self.max_glitch]

    def generate_truth(self):
        ''' Generate truth file'''
        self.truth = "trh_"+self.psrn+".txt"
        with open(self.truth, 'w') as f:
            for gi in range(self.max_glitch):
                idx = gi + 1
                f.write('GLEP_%i   %f\n'%(idx, self.pglep[gi]))
                if self.pglf0[gi] != 0:
                    f.write('GLF0_%i   %e\n'%(idx, self.pglf0[gi]))
                if self.pglf1[gi] != 0:
                    f.write('GLF1_%i   %e\n'%(idx, self.pglf1[gi]))
                if self.pglf2[gi] != 0:
                    f.write('GLF2_%i   %e\n'%(idx, self.pglf2[gi]))
                if self.pglf0d[gi] != 0:
                    f.write('GLF0D_%i   %e\n'%(idx, self.pglf0d[gi]))
                if self.pgltd[gi] != 0:
                    f.write('GLTD_%i   %f\n'%(idx, self.pgltd[gi]))
                if self.pglf0d2[gi] != 0:
                    f.write('GLF0D2_%i   %e\n'%(idx, self.pglf0d2[gi]))
                if self.pgltd2[gi] != 0:
                    f.write('GLTD2_%i   %f\n'%(idx, self.pgltd2[gi]))
                if self.pglf0d3[gi] != 0:
                    f.write('GLF0D3_%i   %e\n'%(idx, self.pglf0d3[gi]))
                if self.pgltd3[gi] != 0:
                    f.write('GLTD3_%i   %f\n'%(idx, self.pgltd3[gi]))
                if self.pglf0[gi] != 0 and self.pglf1[gi] != 0:
                    #glf0_i = glf0 + glf0d + glf0d2 + glf0d3
                    #glf0_T = glf1*t200*86400+glf0d*(np.exp(-t200/gltd)-1)+glf0d2*(np.exp(-t200/gltd2)-1)+glf0d3*(np.exp(-t200/gltd3)-1)
                    f.write('GLF0(instant)_%i   %e\n'%(idx, self.pglf0ins[gi]))
                    f.write('GLF0(T=%d)_%i   %e\n'%(self.taug[gi], idx, self.pglf0tg[gi]))
            if all(p is not None for p in [self.redamp, self.redgam]):
                #alpha = redgam
                #P0 = ((redamp**2)/(12*np.pi**2))*(fc**(-alpha))
                f.write('TNRedAmp   %f\n'%self.redamp)
                f.write('TNRedGam   %f\n'%self.redgam)

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
                        glitches[i] = {'turns':0}
                    if param == "GLEP":
                        glitches[i]['epoch'] = float(e[1])
                    #if param == "GLPH":
                        #glitches[i]['turns'] = round(float(e[1]))
                        #e[1] = "{}".format(float(e[1]) - glitches[i]['turns'])
                    glitches[i][param] = " ".join(e[1:])
                else:
                    parlines.append(line)
        #for ig in glitches:
            #print("glitch[{}] epoch {} turns {}".format(ig,glitches[ig]['epoch'],glitches[ig]['turns']))
        gg = sorted(glitches, key=lambda x: glitches[x]['epoch'])
        for ig in gg:
            if "GLTD" in glitches[ig]:
                if "GLTD2" in glitches[ig]: # Swap 1st and 2nd recoveries if the 1st longer than 2nd
                    if glitches[ig]['GLTD'] > glitches[ig]['GLTD2']:
                        glitches[ig]['GLF0D'], glitches[ig]['GLF0D2'] = glitches[ig]['GLF0D2'], glitches[ig]['GLF0D']
                        glitches[ig]['GLTD'], glitches[ig]['GLTD2'] = glitches[ig]['GLTD2'], glitches[ig]['GLTD']
                if "GLTD3" in glitches[ig]: # Swap 1st and 3rd recoveries if the 1st longer than 3rd
                    if glitches[ig]['GLTD'] > glitches[ig]['GLTD3']:
                        glitches[ig]['GLF0D'], glitches[ig]['GLF0D3'] = glitches[ig]['GLF0D3'], glitches[ig]['GLF0D']
                        glitches[ig]['GLTD'], glitches[ig]['GLTD3'] = glitches[ig]['GLTD3'], glitches[ig]['GLTD']
                    if "GLTD2" in glitches[ig] and glitches[ig]['GLTD2'] > glitches[ig]['GLTD3']:
                        glitches[ig]['GLF0D2'], glitches[ig]['GLF0D3'] = glitches[ig]['GLF0D3'], glitches[ig]['GLF0D2']
                        glitches[ig]['GLTD2'], glitches[ig]['GLTD3'] = glitches[ig]['GLTD3'], glitches[ig]['GLTD2']
        self.tidypar = "tdy_"+self.par.split('_', 1)[1]
        with open(self.tidypar,"w") as f1:
            f1.writelines(parlines)
            for ig in gg:
                for param in glitches[ig]:
                    if param in ["epoch","turns"]:
                        continue
                    f1.write("{}_{} {}\n".format(param,glitches[ig]['newid'],glitches[ig][param]))
        with open(self.tim) as f2:
            for line in f2:
                e = line.split()
                if "-pn" in e:
                    epoch = float(e[2])
                    ii = e.index("-pn")
                    pn = int(e[ii+1])
                    for ig in gg:
                        if epoch > glitches[ig]['epoch']:
                            pn -= glitches[ig]['turns']
                    newline = " ".join(e[:ii])+" -pn {} ".format(pn)+(" ".join(e[ii+2:]))
                    if isinstance(chop, int):
                        if self.pglep[0]-chop <= epoch <= self.pglep[-1]+chop:
                            timlines.append(" "+newline+"\n")
                    else:
                        timlines.append(" "+newline+"\n")
                elif len(e)>3:
                    epoch = float(e[3])
                    if isinstance(chop, int):
                        if self.pglep[0]-chop <= epoch <= self.pglep[-1]+chop:
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
        if isinstance(glitchnum, int):
            if 0 < glitchnum <= self.max_glitch:
                startnum = glitchnum - 1
                endnum = glitchnum + 1
                if endnum > self.max_glitch:
                    endnum = -1
        if startnum<0:
            startnum = 0
        if endnum>self.max_glitch:
            endnum = -1
        if startnum == 0:
            startmjds = self.minmjds
        else:
            startmjds = self.pglep[startnum-1]
        startmjds = min(startmjds, self.pglep[glitchnum-1]-1000)
        if endnum == -1:
            endmjds = self.maxmjds
        else:
            endmjds = self.pglep[endnum-1]
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
                    if startmjds <= epoch <= endmjds:
                        timlines.append(" "+newline+"\n")
                elif len(e)>3:
                    epoch = float(e[3])
                    if startmjds <= epoch <= endmjds:
                        timlines.append(line)
                else:
                    timlines.append(line)
        if isinstance(glitchnum, int):
            self.splittim = "chp_"+self.psrn+"_"+str(glitchnum)+".tim"
        else:
            self.splittim = "chp_"+self.psrn+"_"+str(startnum)+":"+str(endnum)+".tim"
        print("Generating split tim {} for glitch {}".format(self.splittim, self.splittim.split(".")[-2].split("_")[-1]))
        with open(self.splittim,"w") as f2:
            f2.writelines(timlines)
        return self.splittim

    def split_par(self, splittim=None, glitchnum=None, recoveries=0, GLF2=False):
        '''Generate new split par file of different models for MCMC fit, move PEPOCH to the center with epoch centre.
           Glitchnum and recoveries specify the number of recovery terms in model for the glitch to be fit'''
        if splittim is None:
            splittim = self.splittim
        if isinstance(glitchnum, int) and 0 < glitchnum <= self.max_glitch:
            pass
        else:
            glitchnum = int(self.splittim.split('_')[-1].split('.')[0].split(':')[0])
        if isinstance(recoveries, int) and 0 <= recoveries <= 3:
            pass
        else:
            recoveries = 0
        if recoveries==0:
            modelname = 'f'
        elif recoveries==1:
            modelname = 'r'
        elif recoveries==2:
            modelname = 'd'
        elif recoveries==3:
            modelname = 't'
        if GLF2 is True:
            self.splitpar = "glf_"+self.psrn+"_"+str(glitchnum)+modelname+".par"
        else:
            self.splitpar = "bst_"+self.psrn+"_"+str(glitchnum)+modelname+".par"
        subprocess.call(["tempo2", "-epoch", "centre", "-nofit", "-f", self.par, splittim, "-outpar", self.splitpar])
        print("Generating split par {} for glitch {} model {}".format(self.splitpar, glitchnum, modelname))
        parlines = []
        with open(self.splitpar) as f1:
            for line in f1:
                if any(line.startswith(ls) for ls in ["GLEP", "GLF0_", "GLF1_", "GLF2_", "GLF0(", "JUMP"]):
                    if line.startswith("GLF2_{} ".format(str(glitchnum))):
                        GLF2=False
                    parlines.append(line)
                elif line.startswith("TN"):
                    continue
                elif any(line.startswith(ls) for ls in ["RA", "DEC", "PX", "PM", "DM", "F0", "F1", "F2"]):
                    e = line.split()
                    if len(e) >= 3:
                        # Turn on tempo2 fitting of these parameters
                        newline = e[0]+'   '+e[1]+'   '+'1'+'   '+e[-1]+'\n'
                        parlines.append(newline)
                    else:
                        newline = e[0]+'   '+e[1]+'   '+'1'+'\n'
                        parlines.append(newline)
                elif line.startswith("GLF0D_") or line.startswith("GLTD_"):
                    e = line.split()
                    num = int(e[0].split('_')[-1])
                    if num==glitchnum and recoveries < 1:
                        pass
                    else:
                        parlines.append(line)
                elif line.startswith("GLF0D2_") or line.startswith("GLTD2_"):
                    e = line.split()
                    num = int(e[0].split('_')[-1])
                    if num==glitchnum and recoveries < 2:
                        pass
                    else:
                        parlines.append(line)
                elif line.startswith("GLF0D3_") or line.startswith("GLTD3_"):
                    e = line.split()
                    num = int(e[0].split('_')[-1])
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
                            newline = 'GLF0D_'+str(glitchnum)+'   '+'0'+'\n'
                            parlines.append(newline)
                            newline = 'GLTD_'+str(glitchnum)+'   '+str(50*(4**expnum))+'\n'
                            parlines.append(newline)
                        else:
                            newline = 'GLF0D{}_'.format(str(expnum+1))+str(glitchnum)+'   '+'0'+'\n'
                            parlines.append(newline)
                            newline = 'GLTD{}_'.format(str(expnum+1))+str(glitchnum)+'   '+str(50*(4**expnum))+'\n'
                            parlines.append(newline)
                        expterms += 1
            if GLF2 is True:
                newline = 'GLF2_{}'.format(str(glitchnum))+'   '+'0'+'\n'
                parlines.append(newline)
        with open(self.splitpar, "w") as f2:
            f2.writelines(parlines)
        return self.splitpar

    def MCMC_fit(self, par=None, tim=None, glitchnum=1, solver="multinest", recoveries=0, GLF2=False, sigma=[100, 100, 100, 100], gleprange=50, gltdrange=[1, 3], gltdsplit=[2.0, 2.3], glf0drange=0.8, glf0range=0.8, glf1range=0.8, glf2range=10, red=[-16, -8], tspan=1.1, thread=16, N=8000, nwalkers=128, nlive=500, rootdir="/nvme1/yliu/yangliu/"):
        '''Call enterprise to do MCMC fit for model selection'''
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
            glitchnum = int(self.splittim.split('_')[-1].split('.')[0].split(':')[0])
        if isinstance(recoveries, int) and 0 <= recoveries <= 3:
            pass
        else:
            recoveries = 0
        if recoveries==0:
            modelname = 'f'
        elif recoveries==1:
            modelname = 'r'
        elif recoveries==2:
            modelname = 'd'
        elif recoveries==3:
            modelname = 't'
            gltdsplit[0]=1.8
        if GLF2 is True:
            gleprange *= 0.04
        else:
            glf2range = 0
        if glitchnum is not None:
            plotname = str(glitchnum)+modelname
        else:
            plotname = "sum"
        dirname = self.psrn+"_"+plotname
        outputfile = "opt_"+dirname+".txt"
        taug = [str(int(tg)) for tg in self.taug]
        command = [runentprise, "-t", str(thread), "--auto-add"]
        command.extend([par, tim, "--outdir", dirname, "--plotname", plotname])
        if glitchnum is not None:
            command.extend(["--glitches", str(glitchnum)])
        plotchain = "--plot-chain"
        if solver=="emcee":
            command.extend(["-N", str(N), "--nwalkers", str(nwalkers)])
        elif solver=="dynesty":
            plotchain = "--dynesty-plots"
            command.extend(["-nlive", str(nlive)])
        else:
            solver = "multinest"
        print("######")
        print("Start MCMC fitting")
        print("taug is:", self.taug)
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
            opt.write(line.decode('utf-8'))

    def post_par(self, par=None, tim=None, glitchnum=1, recoveries=0):
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
            glitchnum = int(self.splittim.split('_')[-1].split('.')[0].split(':')[0])
        if isinstance(recoveries, int) and 0 <= recoveries <= 3:
            pass
        else:
            recoveries = 0
        if recoveries==0:
            modelname = 'f'
        elif recoveries==1:
            modelname = 'r'
        elif recoveries==2:
            modelname = 'd'
        elif recoveries==3:
            modelname = 't'
        postpar = par+".post"
        self.fnlpar = "fnl_"+par.split("_", 1)[1]
        subprocess.call(["tempo2", "-f", postpar, tim, "-outpar", self.fnlpar])
        if glitchnum is not None:
            print("Generating final par {} for glitch {} model {}".format(self.fnlpar, glitchnum, modelname))
        else:
            print("Generating final par {} for glitch summary".format(self.fnlpar))    
        return self.fnlpar

    def best_model(self, glitchnum=1, recoveries=None, GLF2=False):
        '''Find the best model par for glitch glitchnum based on lnZ.
           Or specify the best model manually with the number of recoveries'''
        if isinstance(glitchnum, int) and 0 < glitchnum <= self.max_glitch:
            pass
        else:
            glitchnum = int(self.splittim.split('_')[-1].split('.')[0].split(':')[0])
        if GLF2 is True:
            prefix = "glf_"
        else:
            prefix = "bst_"
        rootdir = os.getcwd()
        glnum = "_"+str(glitchnum)
        postfix = ".par.post"
        maxlnZ = -np.inf
        bestmodelpar = None
        bestmodel = None
        if isinstance(recoveries, int) and 0 <= recoveries <= 3:
            if recoveries==0:
                model = 'f'
            elif recoveries==1:
                model = 'r'
            elif recoveries==2:
                model = 'd'
            elif recoveries==3:
                model = 't'
            postname = prefix+self.psrn+glnum+model+postfix
            postpar = os.path.join(rootdir, postname)
            if os.path.exists(postpar):
                ev_file = os.path.join(rootdir,"{}_{}{}".format(self.psrn, glitchnum, model), "pmn-stats.dat")
                if os.path.exists(ev_file):
                    with open(ev_file) as f:
                        line = f.readline()
                        line = f.readline()
                        lnev = float(line.split()[5])
                        maxlnZ = lnev
                else:
                    print(">>> Missing evidence file {} <<<".format(ev_file))
                bestmodelpar = postpar
                bestmodel = model
        else: 
            for model in ["f", "r", "d", "t"]:
                postname = prefix+self.psrn+glnum+model+postfix
                postpar = os.path.join(rootdir, postname)
                if os.path.exists(postpar):
                    print("Found results for glitch {} model {} (GLF2: {})".format(glitchnum, model, GLF2))
                    ev_file = os.path.join(rootdir,"{}_{}{}".format(self.psrn, glitchnum, model), "pmn-stats.dat")
                    if os.path.exists(ev_file):
                        with open(ev_file) as f:
                            line = f.readline()
                            line = f.readline()
                            lnev = float(line.split()[5])
                            if lnev > maxlnZ:
                                if lnev < maxlnZ+2.5:
                                    print(">>> Evidence Warning <<<")
                                    print("<<< Glitch {} model {} has similar evidence >>>".format(glitchnum, model))
                                else:
                                    maxlnZ = lnev
                                    bestmodelpar = postpar
                                    bestmodel = model
                    else:
                        print(">>> Missing evidence file {} <<<".format(ev_file))
        print("### Best model par is:", bestmodelpar)
        if bestmodelpar is None:
            print(">>> Missing post par for best model <<<")
        else:
            print("<<< The best model for glitch {} is model {} (GLF2: {}) >>>".format(glitchnum, bestmodel, GLF2))
        return bestmodelpar

    def final_par(self, largeglitch=[1], recoveries=[None]*10, GLF2=False):
        '''Merge glitch parameters in the best model of each glitches to create final par for the pulsar'''
        self.sumpar = "sum_"+self.psrn+".par"
        parlines = []
        with open(self.par) as f1:
            for line in f1:
                if line.startswith("GL"):
                    e = line.split()
                    glnum = int(e[0].split("_")[-1])
                    if glnum in largeglitch:
                        pass
                    else:
                        if line.startswith("GLEP"):
                            newline = e[0]+'   '+e[1]+'\n'
                        elif line.startswith("GLPH"):
                            newline = '#'+line
                        else: # Turn on tempo2 fitting of all other glitch parameters for small glitches
                            if len(e) >= 3:
                                newline = e[0]+'   '+e[1]+'   '+'1'+'   '+e[-1]+'\n'
                            else:
                                newline = e[0]+'   '+e[1]+'   '+'1'+'\n'
                        parlines.append(newline)
                elif line.startswith("TN"):
                    continue
                elif any(line.startswith(ls) for ls in ["RA", "DEC", "PX", "PM", "DM", "F0", "F1", "F2"]):
                    parlines.append(line)
                else:
                    parlines.append(line)
        idx = 0
        for glnum in largeglitch:
            if idx<len(recoveries):
                bmpar = self.best_model(glitchnum=glnum, recoveries=recoveries[idx], GLF2=GLF2)
            else:
                bmpar = self.best_model(glitchnum=glnum, recoveries=None, GLF2=GLF2)
            with open(bmpar) as f:
                for line in f:
                    if any(line.startswith(ls) for ls in ["GLEP", "GLF0_", "GLF1_", "GLF2_", "GLF0D", "GLTD"]):
                        e = line.split()
                        if glnum == int(e[0].split("_")[-1]):
                            parlines.append(line)
            idx += 1
        with open(self.sumpar, "w") as f2:
            f2.writelines(parlines)
        self.MCMC_fit(par=self.sumpar, tim=self.tim, glitchnum=None, GLF2=GLF2)
        self.par = self.post_par(par=self.sumpar, tim=self.tim, glitchnum=None)
        return self.par

    def noglitch_par(self):
        ''' Make a copy of par file without glitch parameters'''
        parlines = []
        print("### Final par for test par is:", self.par)
        with open(self.par) as f:
            for line in f:
                if line.startswith("GLEP_"):
                    parlines.append(line)
                elif line.startswith("GL") or line.startswith("TN"):
                    continue
                elif any(line.startswith(ls) for ls in ["RA", "DEC", "JUMP", "DM", "PM", "PX", "F0", "F1", "F2"]):
                    parlines.append(line)
                else:
                    e = line.split()
                    if len(e) > 3 and float(e[2]) == 1:
                        # Turn off tempo2 fitting of all other parameters
                        newline = e[0]+'   '+e[1]+'   '+e[3]+'\n'
                        parlines.append(newline)
                    elif len(e) == 3 and float(e[2]) == 1:
                        newline = e[0]+'   '+e[1]+'\n'
                        parlines.append(newline)
                    else:
                        parlines.append(line)
        self.testpar = "tst_"+self.par.split('_', 1)[1]
        with open(self.testpar, "w") as newf:
            newf.writelines(parlines)

    def sf_create_global(self, leading, trailing, width, step):
        ''' Set fit timespan using a global par file'''
        try:
            os.remove("global.par")
        except OSError:
            pass
        with open("global.par", 'a') as glob:
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
        command = ['tempo2', '-f', self.testpar, self.tim, '-nofit', '-global', 'global.par',
                   '-fit', 'F0', '-fit', 'F1', '-fit', 'F2', '-epoch', epoch]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
        while proc.poll() is None:
            line = proc.stdout.readline().decode('utf-8')
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
        print('Stride fitting with box width %d and step size %d'%(width, step))
        first, last = self.minmjds, self.maxmjds
        leading = first
        trailing = first + width
        counter = 0
        self.epochfile = self.psrn+'_g'+str(self.max_glitch)+'_w'+str(int(width))+'_s'+str(int(step))+'_epoch.txt'
        with open(self.epochfile, 'w') as f1:
            while trailing <= last:
                leading, trailing = self.sf_create_global(leading, trailing, width, step)
                epoch = leading + ((trailing - leading)/2.0)
                print(leading, trailing, epoch, file=f1)
                counter += 1
                leading = first + counter*step
                trailing = first + width + counter*step
        starts, ends, fitepochs = np.loadtxt(self.epochfile, unpack=True)
        self.datafile = self.psrn+'_g'+str(self.max_glitch)+'_w'+str(int(width))+'_s'+str(int(step))+'_data.txt'
        with open(self.datafile, 'w') as f2:
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
        p1y = (f0 - self.F0 - sff1 - sff2)
        p2y = (f0 - self.F0 - sff1 - sff2 - sfglf0 - sfglf1 - sfglf2)
        p3y = (f0 - self.F0 - sff1 - sff2 - sfglf0 - sfglf1 - sfglf2 - sfexp1 - sfexp2 - sfexp3)
        p4y = (f1 - self.F1) #- sfdf2)
        p5y = (f1 - self.F1 - sfdf2 - sfdglf1 - sfdglf2 + sfdexp)
        if save:
            with open("panel1_{}.txt".format(self.psrn), "w") as file1:
                for i, value in enumerate(sft):
                    file1.write('%f   %e   %e   \n' % (value, 1e6*(p1y[i]), 1e6*f0e[i]))
                file1.close()
            with open("panel2_{}.txt".format(self.psrn), "w") as file2:
                for i, value in enumerate(sft):
                    file2.write('%f   %e   %e   \n' % (value, 1e6*(p2y[i]), 1e6*f0e[i]))
                file2.close()
            with open("panel3_{}.txt".format(self.psrn), "w") as file3:
                for i, value in enumerate(sft):
                    file3.write('%f   %e   %e   \n' % (value, 1e6*(p3y[i]), 1e6*f0e[i]))
                file3.close()
            with open("panel4_{}.txt".format(self.psrn), "w") as file4:
                for i, value in enumerate(sft):
                    file4.write('%f   %e   %e   \n' % (value, 1e10*(p4y[i]), 1e10*f1e[i]))
                file4.close()
            with open("panel5_{}.txt".format(self.psrn), "w") as file5:
                for i, value in enumerate(sft):
                    file5.write('%f   %e   %e   \n' % (value, 1e10*(p5y[i]), 1e10*f1e[i]))
                file5.close()
        if plot:
            plt.errorbar(sft, 1e6*p1y, yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
            plt.show()
            plt.errorbar(sft, 1e6*p2y, yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
            plt.show()
            plt.errorbar(sft, 1e6*p3y, yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
            plt.show()
            plt.errorbar(sft, 1e10*p4y, yerr=1e10*f1e, marker='.', color='k', ecolor='k', linestyle='None')
            plt.show()
            plt.errorbar(sft, 1e10*p5y, yerr=1e10*f1e, marker='.', color='k', ecolor='k', linestyle='None')
            plt.show()

    def print_info(self, index=None):
        '''Print basic info of pulsar'''
        print("")
        print("Parameters in par file")
        print("Pulsar name:", self.psrn)
        print("Period epoch:", self.pepoch)
        print("Cadence", self.cadence)
        print("TOA length:", self.toanum)
        print("F0:", self.F0)
        print("F1:", self.F1)
        print("F2:", self.F2)
        print("")
        self.print_glitch_info(index)

    def print_glitch_info(self, index=None):
        '''Print basic info of the index glitch in the pulsar'''
        if isinstance(index, int) and 0 < index <= self.max_glitch:
            print("The {} glitch".format(index))
            print("Glitch epoch:", self.pglep[index-1])
            print("GLF0:", self.pglf0[index-1])
            print("GLF1:", self.pglf1[index-1])
            print("GLF2:", self.pglf2[index-1])
            print("GLF0D:", self.pglf0d[index-1], " - GLTD", self.pgltd[index-1])
            print("GLF0D2:", self.pglf0d2[index-1], " - GLTD2", self.pgltd2[index-1])
            print("GLF0D3:", self.pglf0d3[index-1], " - GLTD3", self.pgltd3[index-1])
            print("Initial jump (GLFO(instant)_1):", self.pglf0ins[index-1])
            print("Decay jump (GLFO(tau_g)_1):", self.pglf0tg[index-1])
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
            glf0[x > 0] += self.pglf0[gn]
            glf1[x > 0] += self.pglf1[gn] * x[x > 0]
            glf2[x > 0] += 0.5 * self.pglf2[gn] * x[x > 0]**2
            exp1[x > 0] += self.pglf0d[gn] * np.exp(-x[x > 0] / (self.pgltd[gn]*86400.0))
            exp2[x > 0] += self.pglf0d2[gn] * np.exp(-x[x > 0] / (self.pgltd2[gn]*86400.0))
            exp3[x > 0] += self.pglf0d3[gn] * np.exp(-x[x > 0] / (self.pgltd3[gn]*86400.0))
            dglf1[x > 0] += self.pglf1[gn]
            dglf2[x > 0] += self.pglf2[gn] * x[x > 0]
            dexp[x > 0] += exp1[x > 0] / (self.pgltd[gn]*86400) + exp2[x > 0] / (self.pgltd2[gn]*86400) + exp3[x > 0] / (self.pgltd3[gn]*86400)
        else:
            for i in range(self.max_glitch):
                glep = self.pglep[i]
                x = mjd2sec(t, glep)
                glf0[x > 0] += self.pglf0[i]
                glf1[x > 0] += self.pglf1[i] * x[x > 0]
                glf2[x > 0] += 0.5 * self.pglf2[i] * x[x > 0]**2
                exp1[x > 0] += self.pglf0d[i] * np.exp(-x[x > 0] / (self.pgltd[i]*86400.0))
                exp2[x > 0] += self.pglf0d2[i] * np.exp(-x[x > 0] / (self.pgltd2[i]*86400.0))
                exp3[x > 0] += self.pglf0d3[i] * np.exp(-x[x > 0] / (self.pgltd3[i]*86400.0))
                dglf1[x > 0] += self.pglf1[i]
                dglf2[x > 0] += self.pglf2[i] * x[x > 0]
                dexp[x > 0] += exp1[x > 0] / (self.pgltd[i]*86400) + exp2[x > 0] / (self.pgltd2[i]*86400) + exp3[x > 0] / (self.pgltd3[i]*86400)
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
        cosidx=lab[1]=='param_red_cos'
        sinidx=lab[1]=='param_red_sin'
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
            if self.pglf0d[i] > 0:
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
