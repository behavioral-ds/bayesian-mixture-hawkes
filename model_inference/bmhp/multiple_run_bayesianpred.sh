#!/bin/bash

for i in {1..20}
do
    qsub -N "xyregexpt-pred-$i" run_bayesianpred.sh
    sleep 60
done
