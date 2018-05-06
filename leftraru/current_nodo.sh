#!/bin/bash
seeds="0"

im=2+_8
feat="2+_8_rpr"
type_im=2+
clf="xgboost"

#im=${1}
#feat=${2}
#type_im=${3}
#clf=${4}

data_dir=/home/rpezoa/ihc/experiment_data/
#out_dir=output/${feat}_halton_patches/
out_dir=output/${feat}_halton_patches/
b_path=${data_dir}big_${type_im}/features/${im}_rpr.npy
bt_path=${data_dir}big_${type_im}/labels/${im}.npy
f_path=${data_dir}${out_dir}feat_vectors/
t_path=${data_dir}${out_dir}target_vectors/

echo ${data_dir}${sub_dir_out}
for s in $seeds;do

module load python/3.5.2

python3 ../scripts/classification.py ${data_dir}/big_${type_im}/ ${data_dir}${out_dir}${clf} ${im}.tif  --seed ${s}  -is 1000 -bp ${b_path}  -btp ${bt_path} -tp ${t_path}${s}.npy -feat_path ${f_path}${s}.npy -clf ${clf}

done

~             
