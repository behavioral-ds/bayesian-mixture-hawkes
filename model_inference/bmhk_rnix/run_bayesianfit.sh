#!/bin/bash

#PBS
#PBS -l ncpus=4
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
#PBS -l jobfs=32GB
#PBS -l storage=gdata/dk92+gdata/gh47
#PBS -q normal
#PBS -r y
#PBS -l wd

module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-data-analysis/2024.01
conda activate /home/587/pc3426/.conda/envs/stan
export PATH=/home/587/pc3426/.conda/envs/stan/bin:$PATH

BASE="/g/data/gh47/pc3426/jobs/bayesian_mixture_theta_rnix_onlyt_2024/"

for i in {0,}
do
    for j in 2 8 13 26 22 18 11 12 14 21 10 23 24 25 5 9 15 16 1 27 20 19 4 0 17
    do 
        for k in 0 1 2 3
        do
            FILE="${BASE}/tracker/${i}_${j}_${k}.p"

            if [ -f "$FILE" ]
            then
                echo "$FILE is processing / already processed."
            else
                touch $FILE

                SCRATCH="/scratch/gh47/pc3426/bayesian_mixture_theta_rnix_2024_${i}_${j}_${k}"
                if [ ! -d "${SCRATCH}" ]
                then
                    mkdir -p "${SCRATCH}"
                    mkdir "${SCRATCH}/data"
                    mkdir "${SCRATCH}/output"
                    mkdir "${SCRATCH}/stan_files"
                    mkdir "${SCRATCH}/log"
                fi

                cd ${PBS_O_WORKDIR}
                cp data/* "${SCRATCH}/data"
                cp stan_files/* "${SCRATCH}/stan_files"

                cd ${SCRATCH}
                python ${PBS_O_WORKDIR}/run_fit.py $i $j $k

                cp -r "${SCRATCH}/output/"* "${BASE}/output/"
                rm -rf ${SCRATCH}

                cd ${PBS_O_WORKDIR}
            fi
        done
    done
done
