#!/bin/bash
im=2+_1
type_im=2+
s=0
feat=rpr
tmask=0.334
tmax=0.15
patch_num=1

data_dir=/home/rpezoa/experiment_data/
sub_dir=output/
sub_dir_out=output/abril_2018/
dir=${feat}_patches_${tmask}_${tmax}


python3 svm_patches.py ${data_dir}/big_${type_im}/ ${data_dir}${sub_dir_out} ${im}.tif --suffix $dir --seed ${s}  -feat_path ${data_dir}${sub_dir}${im}/${dir}/halton_patches/feat_vectors/${s}_patches_${patch_num}.npy --target_path ${data_dir}${sub_dir}${im}/${dir}/halton_patches/target_vectors/${s}_target_${patch_num}.npy -is 1000 -bp ${data_dir}/big_${type_im}/features/${im}_${feat}.npy  -btp ${data_dir}${sub_dir}${im}/${dir}/halton_patches/big_target_vector.npy

