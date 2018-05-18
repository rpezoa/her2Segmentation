#!/bin/bash
#PBS -q gpuk
#PBS -N log_${clf}_${seed}
#PBS -l walltime=3:00:00

cd $PBS_O_WORKDIR
pwd

im=${im}
type_im=${type_im}
clf=${clf}
type_training_data=${training}
feat=${feat}
under_sampling=${under_sampling}
s=${seed}
cluster=${cluster}
local_rw=${rw}

data_dir=/data/user/r/rpezoa/experiment_data/
out_dir=output/${im}_${feat}_${type_training_data}/
b_path=${data_dir}big_${type_im}/features/${im}_rpr.npy
bt_path=${data_dir}big_${type_im}/labels/${im}.npy
f_path=${data_dir}${out_dir}feat_vectors/
t_path=${data_dir}${out_dir}target_vectors/

use anaconda3
gcc48

python ../scripts/classification.py ${data_dir}/big_${type_im}/ ${data_dir}${out_dir}${clf}_${under_sampling} ${im}.tif  --seed ${s}  -is 1000 -bp ${b_path}  -btp ${bt_path} -tp ${t_path}${s}.npy -feat_path ${f_path}${s}.npy -clf ${clf} -us ${under_sampling} --cluster ${cluster} --local_rw ${local_rw}




