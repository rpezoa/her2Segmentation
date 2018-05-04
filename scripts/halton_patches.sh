seeds="0 1 2 3 4 5 6 7 8 9"
channel_name=blue
c_n=b
kpatches=5
number_patches=100
tmax=0.15 #

#im=2+_8
#feat="2+_8_rpr"
#type_im=2+
#tmask=0.495 # Using an average value!

im=$1
feat=$2
type_im=$3
tmask=$4



data_dir="/home/rpezoa/experiment_data/"
out_dir="output/${feat}_halton_patches/"

f_path=${data_dir}big_${type_im}/features/${feat}.npy
bt_path="/home/rpezoa/experiment_data/big_${type_im}/labels/${im}.npy"
channel_path="/home/rpezoa/experiment_data/big_${type_im}/features/${im}_b.npy"

for s in $seeds;do
    python halton_patches.py ${data_dir}big_${type_im}/ ${data_dir}${out_dir} ${im}.tif --seed $s -feat_path ${f_path} -s 25 -btp ${bt_path} -np ${number_patches} -kp ${kpatches} -is 1000 -th ${tmask} --channel_path ${channel_path} --channel_name ${channel_name} -t_max $tmax
done
~           
