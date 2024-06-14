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
module avail NCI-data-analysis
module load NCI-data-analysis/2021.06
source /g/data/dk92/apps/anaconda3/2020.12/etc/profile.d/conda.sh
conda activate /home/587/pc3426/.conda/envs/stan

BASE="/g/data/gh47/pc3426/jobs/bayesian_mixture_rnix_yregexpt_x/"

for i in {0..1}
do
    for j in {0..50}
    do 
        for k in {0..3}
        do
            FILE="${BASE}/tracker_pred/${i}_${j}_${k}.p"

            if [ -f "$FILE" ]
            then
                echo "$FILE is processing / already processed."
            else
                touch $FILE

                SCRATCH="/scratch/gh47/pc3426/xyregexpt_bayesian_mixture_rnix_pred_${i}_${j}_${k}"
                if [ ! -d "${SCRATCH}" ]
                then
                    mkdir -p "${SCRATCH}"
                    mkdir "${SCRATCH}/data"
                    mkdir -p "${SCRATCH}/output/${i}/${j}/${k}"
                    mkdir "${SCRATCH}/stan_files"
                    mkdir "${SCRATCH}/log"
                    mkdir -p "${SCRATCH}/preds/${i}/${j}/${k}"
                fi

                cd ${PBS_O_WORKDIR}
                cp data/* "${SCRATCH}/data"
                cp "output/${i}/${j}/${k}/"* "${SCRATCH}/output/${i}/${j}/${k}"
                cp stan_files/* "${SCRATCH}/stan_files"

                cd ${SCRATCH}
                python ${PBS_O_WORKDIR}/run_pred.py $i $j $k

                cp -r "${SCRATCH}/preds/"* "${BASE}/preds/"
                rm -rf ${SCRATCH}

                cd ${PBS_O_WORKDIR}
            fi
        done
    done
done
