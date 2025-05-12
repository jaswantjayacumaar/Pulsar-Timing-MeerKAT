#!/bin/bash

par=$1
tim=$2
new=$3

tempo2 -f $par $tim -writeres

ss=$(grep -e "^START" $par | awk '{print $2}')
ff=$(grep -e "^FINISH" $par | awk '{print $2+1000}')

echo ~/Code/run_enterprise/get_gp_timeseries.py -s $ss -f $ff -I
~/Code/run_enterprise/get_gp_timeseries.py -s $ss -f $ff -I -N 100

cat $par cm.ifunc | grep -v "TRACK" | grep -v TNRed > white.par




tempo2 -output add_pulseNumber -f white.par $new
