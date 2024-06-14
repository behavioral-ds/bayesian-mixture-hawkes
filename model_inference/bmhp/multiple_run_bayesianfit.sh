#!/bin/bash

for i in {1..20}
do
    qsub -N "xyregexpt_rnix-$i" run_bayesianfit.sh
    sleep 60
done
