import os, sys
import numpy as np
code = sys.argv[0]
data = sys.argv[1]

psrs = np.loadtxt(data, dtype=str)

for psr in psrs:

	parfile = psr+'/'+'pf_'+psr+'.par'
	timfile = psr+'/'+'pf_'+psr+'.tim'

	com = 'tempo2 -f '+parfile+' '+timfile+' -fit f0 -fit f1 -newpar'
	com2 = 'mv new.par final_'+psr+'.par' 
	com3 = 'mv final_'+psr+'.par '+psr

	os.system(com)
	os.system(com2)
	os.system(com3)
