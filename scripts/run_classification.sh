#!/bin/bash
seeds="0" #1 2 3 4 5 6 7 8 9"

#im=2+_8
#feat="2+_8_rpr"
#type_im=2+
#clf="svm"

im=${1}
feat=${2}
type_im=${3}
clf=${4}
under_sampling=${5}

data_dir=/home/rpezoa/experiment_data/
#out_dir=output/${feat}_halton_patches/
out_dir=output/${feat}_halton_patches/
b_path=${data_dir}big_${type_im}/features/${im}_rpr.npy
bt_path=${data_dir}big_${type_im}/labels/${im}.npy
f_path=${data_dir}${out_dir}feat_vectors/
t_path=${data_dir}${out_dir}target_vectors/

for s in $seeds;do

python3 classification.py ${data_dir}/big_${type_im}/ ${data_dir}${out_dir}${clf}_${under_sampling} ${im}.tif  --seed ${s}  -is 1000 -bp ${b_path}  -btp ${bt_path} -tp ${t_path}${s}.npy -feat_path ${f_path}${s}.npy -clf ${clf} -us ${under_sampling}

done

