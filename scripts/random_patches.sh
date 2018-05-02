seeds="0 1 2 3 4 5 6 7 8 9"
im=2+_1
feat="2+_1_rpr"
type_im=2+
kpatches=5

data_dir="/home/rpezoa/experiment_data/"
out_dir="output/${feat}_random_patches/"
f_path=${data_dir}big_${type_im}/features/${im}_rpr.npy
bt_path="/home/rpezoa/experiment_data/big_${type_im}/labels/${im}.npy"
number_patches=100

for s in $seeds;do
    python random_patches.py ${data_dir}big_${type_im}/ ${data_dir}${out_dir} ${im}.tif  --seed $s -feat_path ${f_path} -s 25 -btp ${bt_path} -np ${number_patches} -kp ${kpatches} -is 1000 
done
~          
~
 
