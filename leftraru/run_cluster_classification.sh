#!/bin/bash
imgs="1+_20 1+_25 2+_1 2+_8 2+_9 3+_19"
types="1+ 1+ 2+ 2+ 2+ 3+"
train="halton_patches random_patches"
feat="rpr"
seeds="0 1 2 3 4 5 6 7 8 9"
seeds="1"

for s in $seeds;do
    sbatch classification.sh 3+_19 3+ svm $s "halton_patches" "rpr" "0" "1" "1"
    #sbatch classification.sh img type_img type_classifier seed 
    #type_training_data feature underSampling running_on_cluster local_readWrite 
done


