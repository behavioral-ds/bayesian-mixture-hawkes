#!/bin/bash

#PBS
#PBS -l ncpus=4
#PBS -l mem=128GB
#PBS -l walltime=48:00:00
#PBS -l jobfs=128GB
#PBS -l storage=gdata/dk92+gdata/gh47
#PBS -q normal
#PBS -r y
#PBS -l wd

module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-data-analysis/2024.01
conda activate /home/587/pc3426/.conda/envs/stan
export PATH=/home/587/pc3426/.conda/envs/stan/bin:$PATH

BASE="/g/data/gh47/pc3426/jobs/bayesian_mixture_theta_fakenix_onlyt_2024/"

for i in {1,}
do
    for j in 28 10 4 29 36 32 7 24 27 35 19 37 30 20 0 11 2 16 33 18 17 6 22 15 14 12 25 21 3 1 26 13 23 31 34
    do 
        for k in 0 1 2 3
        do
            FILE="${BASE}/tracker/${i}_${j}_${k}.p"

            if [ -f "$FILE" ]
            then
                echo "$FILE is processing / already processed."
            else
                touch $FILE

                SCRATCH="/scratch/gh47/pc3426/only_t_2024_bayesian_mixture_theta_fakenix_${i}_${j}_${k}"
                if [ ! -d "${SCRATCH}" ]
                then
                    mkdir -p "${SCRATCH}"
                    mkdir "${SCRATCH}/data"
                    mkdir "${SCRATCH}/output"
                    mkdir "${SCRATCH}/stan_files"
                    mkdir "${SCRATCH}/log"
                fi

                cd ${PBS_O_WORKDIR}
                cp "data/THETA_0_FAKENIX_split_${j}.p" "${SCRATCH}/data"
                cp "data/THETA_1_FAKENIX_split_${j}.p" "${SCRATCH}/data"
                cp "data/THETA_2_FAKENIX_split_${j}.p" "${SCRATCH}/data"

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
