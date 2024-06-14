#!/bin/bash

for i in {1..12}
do
    qsub -N "theta_rnix-$i" run_bayesianfit.sh
    sleep 60
done
