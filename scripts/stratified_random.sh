seeds="0 1 2 3 4 5 6 7 8 9"

#im=2+_8
#feat="2+_8_rpr"
#type_im=2+

im=$1
feat=$2
type_im=$3

data_dir="/home/rpezoa/experiment_data/"
out_dir="output/${feat}_stratified_random/"

f_path=${data_dir}big_${type_im}/features/${feat}.npy
bt_path="/home/rpezoa/experiment_data/big_${type_im}/labels/${im}.npy"

for s in $seeds;do
    python stratified_random.py ${data_dir}big_${type_im}/ ${data_dir}${out_dir} ${im}.tif --seed $s -feat_path ${f_path} -btp ${bt_path}  
done
~           
