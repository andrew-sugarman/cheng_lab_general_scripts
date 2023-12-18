#!/bin/bash
mkdir -p -m2777 slurm_logs

sbatch -p compute --job-name wonderful_ctl -A lab_cheng -D . -o ./slurm_logs/allinwonderful_submission_log.txt ./allinwonderful_v2.54_hpc_springboard.sh '"$(pwd)"'
# sbatch -p compute --job-name wonderful_ctl -D . -o allinwonderful_submission_log.txt /gpfs/Labs/Cheng/phenome/hpc_scripting/reconstruction/allinwonderful_v3_hpc_springboard.sh pwd