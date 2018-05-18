im="2+_8"
type_im="2+"
clf="deep"
s="1"
type_training_data="halton_patches"
feat="rpr"
under_sampling=0
cluster=1

data_dir=/data/user/r/rpezoa/experiment_data/
out_dir=output/${im}_${feat}_${type_training_data}/
b_path=${data_dir}big_${type_im}/features/${im}_rpr.npy
bt_path=${data_dir}big_${type_im}/labels/${im}.npy
f_path=${data_dir}${out_dir}feat_vectors/
t_path=${data_dir}${out_dir}target_vectors/


use anaconda3
use gcc48

python ../scripts/classification.py ${data_dir}/big_${type_im}/ ${data_dir}${out_dir}${clf} ${im}.tif  --seed ${s}  -is 1000 -bp ${b_path}  -btp ${bt_path} -tp ${t_path}${s}.npy -feat_path ${f_path}${s}.npy -clf ${clf} -us ${under_sampling} --cluster ${cluster}


