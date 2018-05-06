#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=slims
#SBATCH -n 20
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive=user
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.error
#SBATCH --mail-user=rpezoarivera@gmail.com
#SBATCH --mail-type=ALL

im=$1
type_im=$2
clf=$3
s=$4
type_training_data=$5
feat=$6

data_dir=/home/rpezoa/ihc/experiment_data/
out_dir=output/${im}_${feat}_${type_training_data}/
b_path=${data_dir}big_${type_im}/features/${im}_rpr.npy
bt_path=${data_dir}big_${type_im}/labels/${im}.npy
f_path=${data_dir}${out_dir}feat_vectors/
t_path=${data_dir}${out_dir}target_vectors/



module load python/3.5.2

python3 ../scripts/classification.py ${data_dir}/big_${type_im}/ ${data_dir}${out_dir}${clf} ${im}.tif  --seed ${s}  -is 1000 -bp ${b_path}  -btp ${bt_path} -tp ${t_path}${s}.npy -feat_path ${f_path}${s}.npy -clf ${clf}


