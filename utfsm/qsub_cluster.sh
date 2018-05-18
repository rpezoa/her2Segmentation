#!/bin/bash
#qsub -v im="2+_8",type_im="2+",clf="deep",training="halton_patches",feat="rpr",under_sampling="0",seed="0",cluster="1" classification.sh
qsub -v im="2+_8",type_im="2+",clf="deep",training="halton_patches",feat="rpr",under_sampling="0",seed="5",cluster="1",rw="2" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="halton_patches",feat="rpr",under_sampling="0",seed="3",cluster="1" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="halton_patches",feat="rpr",under_sampling="0",seed="4",cluster="1" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="halton_patches",feat="rpr",under_sampling="0",seed="5",cluster="1" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="halton_patches",feat="rpr",under_sampling="0",seed="6",cluster="1" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="halton_patches",feat="rpr",under_sampling="0",seed="7",cluster="1" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="halton_patches",feat="rpr",under_sampling="0",seed="8",cluster="1" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="halton_patches",feat="rpr",under_sampling="0",seed="9",cluster="1" classification.sh

#qsub -v im="2+_8",type_im="2+",clf="svm",training="random_patches",feat="rpr",under_sampling="0",seed="8",cluster="1" classification.sh
#qsub -v im="2+_8",type_im="2+",clf="svm",training="random_patches",feat="rpr",under_sampling="0",seed="9",cluster="1" classification.sh
