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
parser.add_argument('-w', '--width', help='Boxcar width (days)', required=True)
parser.add_argument('-s', '--stride', help='Size of stride (days)', required=True)
args = parser.parse_args()
parfile = args.parfile
timfile = args.timfile
width = float(args.width)
stride = float(args.stride)

def get_lims(tim):
    '''
    Gets first and last MJD from tim file
    '''
    mjds = []
    with open(tim, 'r') as timfile:
        for line in timfile.readlines():
            fields = line.split()
            if len(fields) > 2.0 and fields[0] != "C":
                mjds.append(float(fields[2]))
    #print("Range: {} - {}".format(min(mjds),  max(mjds))) 
    return min(mjds), max(mjds)


def create_global(leading, trailing):
    ''' 
    Set fit timespan using a global par file
    '''
    try:
        os.remove("global.par")
    except OSError:
        pass

    with open("global.par", 'a') as glob:
        #if leading < glep and trailing > glep: 
        #    trailing = glep
        #    if trailing - leading < width / 2.0:
        #        leading = glep
        #        trailing = glep + width
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
              '-fit', 'F0', '-fit', 'F1', '-fit', 'F2',
              '-epoch', epoch
              ] 
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    #print(proc.stdout.read())

    while proc.poll() is None:
        l = proc.stdout.readline() 
        fields = l.split()
        if len(fields) > 2.0:
            if fields[0] == "PEPOCH":
                pepoch = fields[3]
            if fields[0] == "F0":
                F0 = fields[3]
                F0_e = fields[4]
            if fields[0] == "F1":
                F1 = fields[3]
                F1_e = fields[4]
    try:    
        return pepoch, F0, F0_e, F1, F1_e
    except UnboundLocalError:
        return None
 

def main():
    first, last = get_lims(timfile)
    leading = first
    trailing = first + width 
    while trailing <= last:
        #print("Fitting between {} and {}".format(leading, trailing))
        leading, trailing = create_global(leading, trailing)
        epoch = leading + ((trailing - leading)/2.0)
        print(leading, trailing, epoch)
        
        #out = run_fit(parfile, timfile, epoch)
        #if out:
        #    print(out[0], out[1], out[2], out[3], out[4], leading, trailing)
        leading+=stride
        trailing+=stride
       


    

if __name__ == "__main__":
    main()

