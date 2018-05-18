#!/bin/bash
seeds="0"

im=3+_19
feat="3+_19_rpr"
type_im=3+
clf="svm"
under_sampling="1"
cluster=1
local_rw=1

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


module load Lmod/6.5
source $LMOD_PROFILE
ml intel/2017a Python/3.6.2


for s in $seeds;do
	python3 ../scripts/classification.py ${data_dir}/big_${type_im}/ ${data_dir}${out_dir}${clf}_${under_sampling} ${im}.tif  --seed ${s}  -is 1000 -bp ${b_path}  -btp ${bt_path} -tp ${t_path}${s}.npy -feat_path ${f_path}${s}.npy -clf ${clf} -us ${under_sampling} --cluster ${cluster} --local_rw ${local_rw}
done

~             
