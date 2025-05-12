#!/usr/bin/env python

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')

import argparse


parser=argparse.ArgumentParser(description="Run 'enterprise' on a single pulsar")
parser.add_argument('par')
parser.add_argument('tim')
parser.add_argument('--f2', type=float, default=0, help='range of f2 to search')
parser.add_argument('-N','--nsample', type=float, default=1e6, help='number of samples in MCMC')
parser.add_argument('-D','--dm',action='store_true', help='Enable DM variation search')
parser.add_argument('--Adm-max',type=float,default=-12,help='Max log10A_DM')
parser.add_argument('--Adm-min',type=float,default=-18,help='Min log10A_DM')
parser.add_argument('--no-red-noise',dest='red',default=True,action='store_false', help='Disable Power Law Red Noise search')
parser.add_argument('--jbo','-j',action='store_true',help='Use -be flag for splitting backends')
parser.add_argument('--be-flag','-f',help='Use specified flag for splitting backends')
parser.add_argument('--Ared-max','-A',type=float,default=-12,help='Max log10A_Red')
parser.add_argument('--Ared-min',type=float,default=-18,help='Min log10A_Red')
parser.add_argument('--red-gamma-max',type=float,default=8,help='Max gamma red')
parser.add_argument('--red-gamma-min',type=float,default=0,help='Min gamma red')
parser.add_argument('--red-prior-log',action='store_true',help='Use uniform prior in log space for red noise amplitude')
parser.add_argument('--red-ncoeff',type=int,default=60,help='Number of red noise coefficients (nC)')
parser.add_argument('-n','--no-sample',dest='sample',default=True,action='store_false', help='Disable the actual sampling...')
parser.add_argument('--no-white',dest='white',default=True,action='store_false', help='Disable efac and equad')
parser.add_argument('--efac-max',type=float, default=5, help='Max for efac prior')
parser.add_argument('--efac-min',type=float, default=0.2, help='Min for efac prior')
parser.add_argument('--ngecorr',action='store_true', help='Add ECORR for the nanograv backends')
parser.add_argument('--white-corner',action='store_true', help='Make the efac/equad corner plots')
parser.add_argument('--all-corner',action='store_true', help='Make corner plots with all params')
parser.add_argument('--pm',action='store_true', help='Fit for PMRA+PMDEC')
parser.add_argument('--px',action='store_true', help='Fit for parallax')
parser.add_argument('--px-range',type=float,default=10, help='Max parallax to search')
parser.add_argument('--px-verbiest',action='store_true', help=argparse.SUPPRESS) #help='Use Verbiest PX prior to correct for L-K bias')
parser.add_argument('--s1400',type=float,default=None, help=argparse.SUPPRESS)  #help='S1400, used for Verbiest PX prior')
parser.add_argument('--pm-angle',action='store_true', help='Fit for PM + angle')
parser.add_argument('--pm-range',type=float,default=10,help='Search range for proper motion (deg/yr)')
parser.add_argument('--models','-M',nargs='+',help='Add a model to model selection stack. Use e.g. --model "-D --f2 --no-red-noise"')
parser.add_argument('--outdir','-o',type=str,help="Output directory for chains etc")
parser.add_argument('--plot-chain',action='store_true', help='Make a plot of the chains')
parser.add_argument('--tspan-mult',type=float,default=2,help='Multiplier for tspan')

args=parser.parse_args()

print(vars(args))

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.stats import rv_continuous


import sys

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
import enterprise.constants as const
from enterprise.signals import deterministic_signals

import enterprise_extensions
from enterprise_extensions import models, model_utils

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from astropy import coordinates as coord
from astropy import units as u

from matplotlib.backends.backend_pdf import PdfPages


par = args.par
tim=args.tim
f2range=args.f2

print("Read pulsar data")
psr=Pulsar(par,tim,drop_t2pulsar=False)

orig_toas=psr.t2pulsar.toas()
issorted=np.all(orig_toas[:-1] <= orig_toas[1:])

psr.earthssb = psr.t2pulsar.earth_ssb
psr.observatory_earth = psr.t2pulsar.observatory_earth

orig_pmra=0
orig_pmdec=0
posepoch=-1
parlines=[]
with open(par) as f:
    for line in f:
        parlines.append(line)
        e=line.split()
        if len(e) > 1:
            if e[0]=="F0":
                f0=float(e[1])
            elif e[0]=="PEPOCH":
                pepoch=float(e[1])
            elif e[0]=="POSEPOCH":
                posepoch=float(e[1])
            elif e[0]=="PMRA":
                orig_pmra=float(e[1])
            elif e[0]=="PMDEC":
                orig_pmdec=float(e[1])
if posepoch < 0:
    print("Assume POSEPOCH=PEPOCH")
    posepoch=pepoch

if args.px and "PX" in psr.fitpars:
    print("ERROR: Can't fit for PX in both MCMC and least-squares")
    sys.exit(1)
if args.pm and "PMRA" in psr.fitpars:
    print("ERROR: Can't fit for PMRA in both MCMC and least-squares")
    sys.exit(1)
if args.pm and "PMDEC" in psr.fitpars:
    print("ERROR: Can't fit for PMDEC in both MCMC and least-squares")
    sys.exit(1)

if (args.pm or args.px) and not issorted:
    print("ERROR: Toas must be sorted or wierd things happen with paralax and proper motion")
    sys.exit(1)

print("Done.")


def jbbackends(flags):
    backend_flags=flags['be']
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def flagbackends(flags,beflagval):
    backend_flags=flags[beflagval]
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


@signal_base.function
def mjkf2(toas,pepoch,f0,valf2):
    x = (toas-pepoch*86400.0)
    return -1e-27*valf2*x*x*x/6.0/f0

@signal_base.function
def mjkpm_angle(toas,pos,earthssb,posepoch,amp,angle,orig_pmra=0,orig_pmdec=0):
    psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))

    pmra = amp * np.cos(angle)/np.cos(psrdec)
    pmdec = amp * np.sin(angle)
    return mjkpm(toas,pos,earthssb,posepoch,pmra,pmdec,orig_pmra,orig_pmdec)
 


@signal_base.function
def mjkpx(toas,pos,earthssb,observatory_earth,px):
    # This is just the linearised parallax function from tempo2...
    pxconv=1.74532925199432958E-2/3600.0e3
    AULTSC=499.00478364

    rca = earthssb[:,0:3] + observatory_earth[:,0:3]
    rr = np.sum(rca*rca,axis=1)

    rcos1 = np.sum(pos*rca,axis=1)

    return px * 0.5*pxconv * (rr-rcos1*rcos1)/AULTSC

"""
class VerbiestParallaxPriorRV(rv_continuous):
    def _rvs(self):
        return self.i_invCDF(np.random.uniform(0,1,self._size))
    def _pdf(self,x):
        if x < self.a or x > self.b:
            return 0
        else:
            return self.i_PDF(x)
    def _logpdf(self,x):
        if x < self.a or x > self.b:
            return -np.inf
        else:
            return np.log(self.i_PDF(x))
    def _cdf(self,x):
        if x < self.a:
            return 0
        elif x > self.b:
            return 1
        else:
            return self.i_CDF(x)

def VerbiestParallaxPrior(gl,gb,PXmin=1e-3,PXmax=15,s1400=None,R0=8.5,EE=0.33,size=None):
    GalCo=np.array([gl,gb])
    EE=0.5
    R0=8.5
    PP=np.linspace(PXmin,PXmax,10000)
    RR = np.sqrt( R0 * R0 + np.power( np.cos( GalCo[0] ) / PP, 2.0 )
                           - 2.0 * R0 * np.cos( GalCo[0] ) * np.cos( GalCo[1] ) / PP );
    Pvol = np.power( RR, 1.9 ) \
               * np.exp( - abs( np.sin( GalCo[0] ) / PP ) / EE \
                  - 5.0 * ( RR - R0 ) / R0 ) * np.power( PP, -4.0 )

    Pvol /= np.max(Pvol)
    zz = Pvol

    if s1400 != None:
        Plum = np.power( PP, -1.0 ) *\
            np.exp( -0.5 * np.power( ( np.log10( s1400 ) - 2.0 * np.log10( PP ) + 1.1 )\
                / 0.9, 2.0 ) );
        Plum/=np.max(Plum)
        zz *= Plum


    PDF = zz/np.sum(zz)
    CDF = integrate.cumtrapz(PDF,initial=0)
    CDF -= np.amin(CDF)
    CDF /= np.amax(CDF)

    RV = VerbiestParallaxPriorRV(a=PXmin,b=PXmax)
    RV.i_PDF    = interpolate.interp1d(PP,PDF)
    RV.i_CDF    = interpolate.interp1d(PP,CDF)
    RV.i_invCDF = interpolate.interp1d(CDF, PP)

    class VerbiestParallaxPrior(parameter.Parameter):
        _prior=prior.Prior(RV)
        _size=size
        _gl=gl
        _gb=gb
        def __repr__(self):
            return '"{}":VerbiestParallax({},{})'.format(self.name, gl,gb) \
                + ('' if self._size is None else '[{}]'.format(self._size))
    return VerbiestParallaxPrior
"""

@signal_base.function
def mjkpm(toas,pos,earthssb,posepoch,pmra,pmdec,orig_pmra=0,orig_pmdec=0):
    # This is just the linearised proper motion function from tempo2...
    AULTSC=499.00478364
    rce = earthssb[:,0:3]
    re = np.sqrt(np.sum(rce*rce,axis=1))
   
    psrra=np.arctan2(pos[1],pos[0])+2*np.pi
    psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))
    
    axy=rce[:,2]/AULTSC
    s = axy/(re/AULTSC)
    deltae = np.arctan2(s,np.sqrt(1.0-s*s))
    alphae = np.arctan2(rce[:,1],rce[:,0])
  
    # Convert radians/second to degrees per year
    # 60.0*60.0*1000.0*86400.0*365.25/24.0/3600.0
    rs2dy = (180.0/np.pi)*1314900000

    # This will be pmra in rad/s
    v_pmra  = (pmra-orig_pmra) / rs2dy / np.cos(psrdec)
    v_pmdec = (pmdec-orig_pmdec) / rs2dy
    
    t0 = (toas/86400.0-posepoch)
    d_pmra = re * np.cos(deltae)*np.cos(psrdec)*np.sin(psrra-alphae) * t0
    d_pmdec = re*(np.cos(deltae)*np.sin(psrdec)*np.cos(psrra-alphae) - np.sin(deltae)*np.cos(psrdec))*t0
    
    return v_pmra*d_pmra + v_pmdec*d_pmdec

@signal_base.function
def powerlaw_nogw(f, log10_A=-16, gamma=5):
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    return ((10**log10_A)**2 *
            const.fyr**(gamma-3) * f**(-gamma) * np.repeat(df, 2))



def FourierBasisGP_DM(spectrum, components=20,
                   selection=Selection(selections.no_selection),
                   Tspan=None, name=''):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = utils.createfourierdesignmatrix_dm(nmodes=components, Tspan=Tspan)
    BaseClass = gp_signals.BasisGP(spectrum, basis, selection=selection, name=name)

    class FourierBasisGP_DM(BaseClass):
        signal_type = 'basis'
        signal_name = 'dm noise'
        signal_id = 'dm_noise_' + name if name else 'dm_noise'

    return FourierBasisGP_DM


def makeModel(args):
    print("Set up model...")

    f2range=args.f2
    selection = selections.Selection(selections.by_backend)
    selflag='-f'
    if args.jbo:
        selection = selections.Selection(jbbackends)
        selflag='-be'
    if args.be_flag:
        selflag="-"+args.be_flag
        selection = selections.Selection(lambda flags: flagbackends(flags,args.be_flag))


    s = selection(psr)



    tm = gp_signals.TimingModel()
    model =  tm

    if args.white:
        medianlogerr=np.log10(np.median(psr.toaerrs))
        efac = parameter.Uniform(args.efac_min,args.efac_max)
        equad = parameter.LinearExp(medianlogerr-2,medianlogerr+2)
        ef = white_signals.MeasurementNoise(efac=efac,selection=selection)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
        model += ef
        model += eq
    else:
        ef = white_signals.MeasurementNoise(efac=parameter.Constant(1.0),selection=selection)
        model += ef

    if args.ngecorr:
        ngselection = selections.Selection(selections.nanograv_backends)
        ecorr = parameter.LinearExp(-9,-5)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ngselection)
        model += ec

    if args.red:
        # red noise (powerlaw with 30 frequencies)
        if args.red_prior_log:
            log10_A = parameter.Uniform(args.Ared_min,args.Ared_max)
        else:
            log10_A = parameter.LinearExp(args.Ared_min,args.Ared_max)
        gamma = parameter.Uniform(args.red_gamma_min,args.red_gamma_max)
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=nC, Tspan=Tspan)
        model+=rn

    if args.dm:
        log10_Adm = parameter.LinearExp(args.Adm_min,args.Adm_max)('DM_A')
        gamma_dm = parameter.Uniform(0,6)('DM_gamma')
        pldm = powerlaw_nogw(log10_A=log10_Adm, gamma=gamma_dm)
        dm = FourierBasisGP_DM(spectrum=pldm, components=nC, Tspan=Tspan)
        model+=dm

    if args.pm:
        pmrange=args.pm_range
        pmra=parameter.Uniform(-pmrange+orig_pmra,pmrange+orig_pmra)
        pmdec=parameter.Uniform(-pmrange+orig_pmdec,pmrange+orig_pmdec)
        pm = deterministic_signals.Deterministic(mjkpm(posepoch=posepoch,pmra=pmra,pmdec=pmdec,orig_pmra=orig_pmra,orig_pmdec=orig_pmdec),name='pm')
        model += pm

    if args.px:
        if args.px_verbiest:
            print("\n\nWarning: Using Verbiest PX prior (Not tested!!)\n\n")
            print("\n\nERROR: FAIL\n\n")
            sys.exit(1)
            """
            if f0 > 100:
                px_ee=0.5
            else:
                px_ee=0.33
            ra=np.arctan2(psr.pos[1],psr.pos[0])*u.radian
            dec=np.arcsin(psr.pos[2]/np.sqrt(np.sum(psr.pos*psr.pos)))*u.radian
            c = coord.SkyCoord(ra,dec)
            gal = c.transform_to(coord.Galactic)
            gl=gal.l.deg
            gb=gal.b.deg
            pxp=VerbiestParallaxPrior(gl,gb,s1400=args.s1400,EE=px_ee)
            """
        else:
            pxp=parameter.Uniform(0,args.px_range)
        px = deterministic_signals.Deterministic(mjkpx(px=pxp),name='px')
        model += px

    if args.pm_angle:
        pmrange=args.pm_range
        pmamp=parameter.Uniform(0,pmrange)
        pmangle=parameter.Uniform(0,2*np.pi)
        pm = deterministic_signals.Deterministic(mjkpm_angle(posepoch=posepoch,amp=pmamp,angle=pmangle),name='pm')
        model += pm


    if f2range > 0:
        f2 = mjkf2(pepoch=parameter.Constant(pepoch),f0=parameter.Constant(f0), valf2=parameter.Uniform(-f2range/1e-27,f2range/1e-27)('F2'))
        f2f = deterministic_signals.Deterministic(f2,name='f2')
        model+= f2f
    return model


nC=args.red_ncoeff
Tspan = psr.toas.max()-psr.toas.min()
print("Tspan=%.1f"%(Tspan/86400.))

Tspan*=args.tspan_mult
nC*=args.tspan_mult


#print(psr.fitpars)


selflag='-f'
if args.jbo:
    selflag='-be'
if args.be_flag:
    selflag="-"+args.be_flag


if args.outdir == None:
    outdir='chains/'+psr.name+"/"
else:
    outdir=args.outdir

print("Output directory =",outdir)


if args.models==None:
    model = makeModel(args)
    pta = signal_base.PTA(model(psr))
    sampler=model_utils.setup_sampler(pta,outdir)
    x0 = np.hstack(p.sample() for p in pta.params)
else:
    nmodels = len(args.models)
    mod_index = np.arange(nmodels)
    pta = dict.fromkeys(mod_index)
    i=0
    for m in args.models:
        margs=[args.par,args.tim]
        margs.extend(m.split())
        if args.jbo:
            margs.append("-j")

        if args.be_flag:
            margs.append("--be-flag")
            margs.append(args.be_flag)
        if args.red_prior_log:
            margs.append("--red-prior-log")

        margs.append("--efac-max")
        margs.append("{}".format(args.efac_max))
        margs.append("--efac-min")
        margs.append("{}".format(args.efac_min))

        pargs=parser.parse_args(margs)
        model=makeModel(pargs)
        pta[i] = signal_base.PTA(model(psr))
        i += 1
    super_model = model_utils.HyperModel(pta)
    sampler = super_model.setup_sampler(resume=False, outdir=outdir)
    x0 = super_model.initial_sample()


print("PSR Name: ",psr.name)
print("F0: ",f0)
print("PEPOCH: ",pepoch)
print("POSPOCH: ",posepoch)
print("")
print("'least-squares' fit parameters:")
i=1
for p in psr.fitpars:
    print("  {:2d}   {}".format(i,p))
    i+=1


pars = np.loadtxt(outdir + '/pars.txt', dtype=np.unicode_)
print("MCMC fit parameters:")
i=1
for p in pars:
    z=""
    if not args.models == None:
        for ii in range(len(args.models)):
            if p in pta[ii].param_names:
                z += "%d"%ii
            else:
                z += "_"


    print("  {:2d}   {:40s} {}".format(i,p,z))
    i+=1


# sampler for N steps
N = int(args.nsample)
if args.sample:
    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)


# In[8]:


chain = np.loadtxt(outdir+'/chain_1.txt')
pars = np.loadtxt(outdir + '/pars.txt', dtype=np.unicode_)
burn = int(0.25 * chain.shape[0])


if not args.models == None:
    msfname=par+".models"
    with open(msfname,"w") as msf:
        # We want to find the best model...
        odd_vals = np.zeros(len(args.models))
        samples = chain[burn:,-5]
        for i in range(len(args.models)):
            mask = np.rint(samples) == i
            odd_vals[i] = float(np.sum(mask))
        bestmodel = np.argmax(odd_vals)
        print("Doing model selection...")
        print("Best Model: {} '{}'".format(bestmodel,args.models[bestmodel]))
        msf.write("Best Model: {} '{}'\n".format(bestmodel,args.models[bestmodel]))
        print("Odds ratio:")
        for i in range(len(args.models)):
            if odd_vals[i] == 0:
                frac = 1.0 / odd_vals[bestmodel]
            else:
                frac = odd_vals[i]/odd_vals[bestmodel]
            
            print("Model P(%d)/P(%d): %.6lf   %.1f"%(i,bestmodel,frac,np.log(frac)))
            msf.write("Model P(%d)/P(%d): %.6lf   %.1f\n"%(i,bestmodel,frac,np.log(frac)))
        #print(model_utils.odds_ratio(chain[burn:,-5],models=[0,1]))

        post = chain[burn:,:len(pars)]

        m_f2 = (pars=='F2')
        i_ef = np.array([i for i, v in enumerate(pars) if 'efac' in v],dtype=np.int)
        i_eq = np.array([i for i, v in enumerate(pars) if 'equad' in v],dtype=np.int)

        m_ef = np.zeros(len(pars),dtype=np.bool)
        m_ef[i_ef] = True
        m_eq = np.zeros(len(pars),dtype=np.bool)
        m_eq[i_eq] = True

        n_w=np.logical_not(np.logical_or(m_ef,m_eq))
        imax = np.argmax(chain[:,len(pars)+1])
        pmax = chain[imax,:len(pars)]

        fig=corner.corner(post[:,n_w], labels=pars[n_w], smooth=True,truths=pmax[n_w])

        fig.savefig("%s.ms_corner.pdf"%psr.name)



        print("\n~~~~~\n")
        print("Now restrict parameters to those in best model...")
        pta = pta[bestmodel]

        pmask=np.zeros(len(pars),dtype=bool)
        for pp  in pta.param_names:
            idx = pars == pp
            pmask = np.logical_or(pmask,idx)
        pars = pars[pmask]
        pmask = np.append(pmask,[True,True,True,True])
        mask = np.rint(chain[:,-5]) == bestmodel
        chain = chain[:,:len(pmask)]
        chain = chain[mask,:]
        chain = chain[:,pmask]
        burn = int(0.25 * chain.shape[0])


imax = np.argmax(chain[:,len(pars)+1])

pmax = chain[imax,:len(pars)]

def convert_f2(v):
    return v*1e-27

def convert_dmA(v):
    v = 10**v
    v *=  (utils.const.DM_K * 1400**2 * 1e12)
    vv = v*v
    v = np.log10(np.sqrt(vv))
    return v


ul68 = np.percentile(chain[burn:,:len(pars)],84.1,axis=0)
ll68 = np.percentile(chain[burn:,:len(pars)],15.9,axis=0)
ul = np.percentile(chain[burn:,:len(pars)],97.5,axis=0)
ll = np.percentile(chain[burn:,:len(pars)],2.5,axis=0)
means = np.mean(chain[burn:,:len(pars)],axis=0)
stds = np.std(chain[burn:,:len(pars)],axis=0)
medians = np.median(chain[burn:,:len(pars)],axis=0)

print("")
print("Max LogLikelihood",pta.get_lnlikelihood(pmax))
print("")
outpar=par+".post"
resfname=par+".results"
with open(resfname,"w") as resf:
    with open(outpar,"w") as outf:
        s=("{:20s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}".format("#param"," max-like"," mean"," std"," 2.5%"," 15.9%"," 50%"," 84.1%"," 97.5%"))

        print(s)
        resf.write(s+"\n")
        for line in parlines:
            outf.write(line)
        for p,v,u,l,mean,median,sig,u68,l68 in zip(pars,pmax,ul,ll,means,medians,stds,ul68,ll68):
            if p=="F2":
                p="F2"
                v = convert_f2(v)
                mean = convert_f2(mean)
                sig = convert_f2(sig)
                l = convert_f2(l)
                l68 = convert_f2(l68)
                median = convert_f2(median)
                u68 = convert_f2(u68)
                u = convert_f2(u)
            elif "efac" in p:
                flag=p.split('_',1)[1]
                flag=flag[:-5]
                p="TNEF %s %s"%(selflag,flag)
            elif "ecorr" in p:
                flag=p.split('_',1)[1]
                flag=flag[:-12]
                p="TNECORR %s %s"%(selflag,flag)
                v = np.pow(10,v)*1e6 # Convert log-seconds to microseconds
            elif "equad" in p:
                flag=p.split('_',1)[1]
                flag=flag[:-12]
                p="TNEQ %s %s"%(selflag,flag)
            elif "pm_angle" in p:
                p="PM-angle"
            elif "pm_amp" in p:
                p="PM"
            elif "px_px" in p:
                p='PX'
            elif "pm_pmra" in p:
                p='PMRA'
            elif "pm_pmdec" in p:
                p='PMDEC'
            elif p=="DM_A":
                v = convert_dmA(v)
                mean = convert_dmA(mean)
                sig = convert_dmA(sig)
                l = convert_dmA(l)
                l68 = convert_dmA(l68)
                median = convert_dmA(median)
                u68 = convert_dmA(u68)
                u = convert_dmA(u)
                outf.write("TNDMC %d\n"%nC)
                p="TNDMAmp"
            elif p=="DM_gamma":
                p="TNDMGam"
            elif "_gamma" in p:
                p="TNRedGam"
            elif "log10_A" in p:
                p="TNRedAmp"
                outf.write("TNRedC %d\n"%nC)
                outf.write("TNRedFLow -0.301\n")
            s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g} ".format(p,v,mean,sig,l,l68,median,u68,u)
            resf.write(s+"\n")
            print(s)
            if p=="PM":
                pmamp=v
                continue
            if p=="PM-angle":
                pmangle=v
                pos=psr.pos
                psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))
                pmra = pmamp * np.cos(pmangle)/np.cos(psrdec)
                pmdec = pmamp * np.sin(pmangle)
                outf.write("PMRA %s\n"%(pmra))
                outf.write("PMDEC %s\n"%(pmdec))
                continue
            outf.write("%s %s\n"%(p,v))

    print("\nSaved par file to '%s'\n"%outpar)


    print("Derived parameters:")
    if psr.name+"_pm_pmra" in pars and psr.name+"_pm_pmdec" in pars:
        pos=psr.pos
        psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))
        i=0
        for p in pars:
            if "pmra" in p:
                pmra = chain[burn:,i]
            if "pmdec" in p:
                pmdec = chain[burn:,i]
            i+=1
        pmz = pmra * np.cos(psrdec)
        pmamp = np.sqrt(pmz*pmz + pmdec*pmdec)
        mean = np.mean(pmamp)
        median=np.median(pmamp)
        u = np.percentile(pmamp,97.5)
        l = np.percentile(pmamp,2.5)
        v = pmamp[imax-burn]
        sig=np.std(pmamp)
        u68 = np.percentile(pmamp,84.1)
        l68 = np.percentile(pmamp,15.9)
        p = "PM"
        s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g} ".format(p,v,mean,sig,l,l68,median,u68,u)
        print(s)
        resf.write(s+"\n")





# In[10]:


post = chain[burn:,:len(pars)]


if args.plot_chain:
    x=np.arange(chain.shape[0])
    with PdfPages("%s.chain.pdf"%psr.name) as pdf:
        for i in range(len(pars)):
            fig=plt.figure(figsize=(16,8))
            plt.plot(x[:burn],chain[:burn,i],'.',color='gray')
            plt.plot(x[burn:],chain[burn:,i],'.',color='k')
            plt.title("%s"%pars[i])
            plt.ylabel("%s"%pars[i])
            pdf.savefig()
            plt.close()



m_f2 = (pars=='F2')
i_ef = np.array([i for i, v in enumerate(pars) if 'efac' in v],dtype=np.int)
i_eq = np.array([i for i, v in enumerate(pars) if 'equad' in v],dtype=np.int)

m_ef = np.zeros(len(pars),dtype=np.bool)
m_ef[i_ef] = True
m_eq = np.zeros(len(pars),dtype=np.bool)
m_eq[i_eq] = True

n_w=np.logical_not(np.logical_or(m_ef,m_eq))
imax = np.argmax(chain[:,len(pars)+1])
pmax = chain[imax,:len(pars)]

fig=corner.corner(post[:,n_w], labels=pars[n_w], smooth=True,truths=pmax[n_w])

fig.savefig("%s.corner.pdf"%psr.name)


if args.all_corner:
    fig=corner.corner(post, labels=pars, smooth=True);
    fig.savefig("%s.corner_all.pdf"%psr.name)



if args.white_corner:
    fig=corner.corner(post[:,m_ef], labels=pars[m_ef], smooth=True);
    fig.savefig("%s.corner_efac.pdf"%psr.name)
    fig=corner.corner(post[:,m_eq], labels=pars[m_eq], smooth=True);
    fig.savefig("%s.corner_equad.pdf"%psr.name)



if f2range > 0:
    print("F2",np.mean(post[:,m_f2])*1e-27,np.std(post[:,m_f2])*1e-27)
