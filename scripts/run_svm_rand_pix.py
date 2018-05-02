#!/bin/bash
im=2+_8
type_im=2+
s=0
feat=har
tmask=0.0
tmax=0.0

data_dir=/home/rpezoa/experiment_data/
sub_dir=output/
sub_dir_out=output/abril_2018_svm_rand_pix/
dir=${feat}_random_
b_path="/home/rpezoa/her2Segmentation/data/2+_8_har_d2_b3.npy"
bt_path="/home/rpezoa/experiment_data/big_2+/labels/2+_8.npy"

python3 svm_rand_pix.py ${data_dir}/big_${type_im}/ ${data_dir}${sub_dir_out} ${im}.tif --suffix $dir --seed ${s}  -is 1000 -bp ${b_path}  -btp ${bt_path}

