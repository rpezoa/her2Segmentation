seeds="0 1 2 3 4 5 6 7 8 9"
im=2+_8
feat="2+_8_rpr"
type_im=2+
kpatches=1

data_dir=/home/rpezoa/experiment_data/
out_dir=output/${feat}_one_rand_patch/
f_path=${data_dir}big_${type_im}/features/${im}_rpr.npy
bt_path=/home/rpezoa/experiment_data/big_2+/labels/2+_8.npy
number_patches=1

for s in $seeds;do
    python one_random_patch.py ${data_dir}big_${type_im}/ ${data_dir}${out_dir} ${im}.tif --seed $s -feat_path ${f_path} -s 57 -btp ${bt_path} -np ${number_patches} -kp ${kpatches} -is 1000 
done
~           
