#!/bin/bash

images=("1+_20" "1+_25" "2+_1" "2+_8" "2+_9" "3+_19")
type_im=("1+" "1+" "2+" "2+" "2+" "3+")
seeds="0 1 2 3 4 5 6 7 8 9"
methods="halton_patches" # random_patches"
images=("1+_20")
type_im=("1+")
seeds="0"
for m in $methods;do
for s in $seeds;do
	for i in `seq 0 0`;do
		echo $m $s ${images[$i]} ${type_im[$i]}
    		qsub -v im=${images[$i]},type_im=${type_im[$i]},clf="deep",training=${m},feat="rpr",under_sampling="0",seed=${s},cluster="1",rw="2" classification.sh
		sleep 1
	done;
done;
done;

#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="0",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="1",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="2",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="3",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="4",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="5",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="6",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="7",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="8",cluster="1",rw="2" classification.sh
#qsub -v im="1+_20",type_im="1+",clf="knn",training="random_patches",feat="rpr",under_sampling="0",seed="9",cluster="1",rw="2" classification.sh
#

