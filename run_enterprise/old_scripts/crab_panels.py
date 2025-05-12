#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

#from matplotlib.ticker import FormatStrFormatter
#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['ps.fonttype'] = 42

par="../test_final.par"

t, f0, f0e, f1, f1e, mjds, mjdf = np.loadtxt("stride_data.txt", unpack=True)

pglep=np.zeros(100)
pglf0=np.zeros(100)
pglf1=np.zeros(100)

pglf0d=np.zeros(100)
pgltd=np.zeros(100)
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
        if e[0].startswith("GLTD_"):
            i=int(e[0][5:])
            pgltd[i-1] = float(e[1])
        if e[0].startswith("GLF0D_"):
            i=int(e[0][6:])
            pglf0d[i-1] = float(e[1])
        if e[0].startswith("GLF1_"):
            i=int(e[0][5:])
            pglf1[i-1] = float(e[1])
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

glep = 58687.565225987409999

def glexp(xx,td,f0d):
    '''
    xx = time since glitch epoch
    td = decay timescale
    f0d = decay amplitude
    '''

    ee = np.zeros_like(f2)
    tau1=td*86400.0
    ee[xx>0] = f0d * np.exp(-xx[xx>0]/tau1)
    return ee


#Time since period epoch
x = (t-58687.5)*86400.0

#Time since glitch epoch
xx = (t-58687.565225987409999)*86400.0

#second derivative term of taylor series
#f = f0 + f1 t + 0.5 f2 t^2
f2 = 0.5 * x * x * F2

#First derivative term of taylor series
f1 = F1 * x

#Permanent change term (constant)
glf0=np.zeros_like(f2)
glf0[t>glep] =  pglf0[0]

#GLF1 term
glf1 = np.zeros_like(xx)
glf1[xx>0] = xx[xx>0] * pglf1[0]

#transient terms
exp1=glexp(xx,pgltd[0],pglf0d[0])
exp2=glexp(xx,pgltd[1],pglf0d[1])
if max_glitch>2:
    exp3=glexp(xx,pgltd[2],pglf0d[2])
else:
    exp3=np.zeros_like(xx)

#plt.errorbar(t, 1e6*(f0 - F0 - f1 - f2 - glf0 - glf1 - exp1), yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
plt.errorbar(t, 1e6*(f0 - F0 - f1 - f2), yerr=1e6*f0e, marker='.', color='k', ecolor='k', linestyle='None')
plt.show()

for i in range(0, len(t)):
    print(t[i], 1e6*(f0[i] - F0 - f1[i] - f2[i] - glf0[i] - glf1[i] - exp1[i]), 1e6*f0e[i])


