#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import argparse
import sys, os
import subprocess
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Stride routine for fitting over a glitch. Written by B.Shaw (benjamin.shaw@manchester.ac.uk.')
parser.add_argument('-p', '--parfile', help='Path to ephemeris', required=True)
parser.add_argument('-t', '--timfile', help='Path to tim file', required=True)
parser.add_argument('-e', '--epochs', help='Path to epochs file', required=True)
args = parser.parse_args()
parfile = args.parfile
timfile = args.timfile
epochs = args.epochs

def create_global(leading, trailing):
    ''' 
    Set fit timespan using a global par file
    '''
    try:
        os.remove("global.par")
    except OSError:
        pass

    with open("global.par", 'a') as glob:
        glob.write("START {} 1\n".format(leading))
        glob.write("FINISH {} 1".format(trailing))

    return leading, trailing



def run_fit(par, tim, epoch):
    '''
    Run tempo2 and fit for parameters
    '''
    epoch = str(epoch)
    command = [
              'tempo2', '-f', par, tim, 
              '-nofit', '-global', 'global.par',  
              #'-fit', 'F0', '-fit', 'F1', '-fit', 'F2',
              '-fit', 'F0', '-fit', 'F1',
              '-epoch', epoch
              ] 
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    #print(proc.stdout.read)
    while proc.poll() is None:
        l = proc.stdout.readline().decode('utf-8')
        fields = l.split()
        if len(fields) > 2:
            if fields[0] == "PEPOCH":
                pepoch = fields[3]
            if fields[0] == "F0":
                F0 = fields[3]
                F0_e = fields[4]
                if float(F0_e)==0 or F0_e=='nan': 
                    return None
            if fields[0] == "F1":
                F1 = fields[3]
                F1_e = fields[4]
                if float(F1_e)==0 or F1_e=='nan': 
                    return None
        try:    
            return pepoch, F0, F0_e, F1, F1_e
        except UnboundLocalError:
            #print("here")
            return None
 

def main():
    starts, ends, fitepochs = np.loadtxt(epochs, unpack=True)
    for i in range(0, len(starts)):
        #print("Fitting between {} and {}".format(starts[i], ends[i]))
        create_global(starts[i], ends[i])
        epoch = fitepochs[i]
        
        out = run_fit(parfile, timfile, epoch)
        os.rename("global.par", str(starts[i]) + "_" + str(ends[i]) + ".eph")
        if out:
            print(out[0], out[1], out[2], out[3], out[4], starts[i], ends[i])
       


    

if __name__ == "__main__":
    main()

