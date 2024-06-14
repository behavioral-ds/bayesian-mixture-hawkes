#!/bin/bash

for i in {13..24}
do
    qsub -N "theta_fakenix-$i" run_bayesianfit.sh
    sleep 60
done
