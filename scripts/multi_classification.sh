#./run_classification.sh "1+_20" "1+_20_rpr" "1+" "svm"
#./run_classification.sh "1+_20" "1+_20_rpr" "1+" "svm"
#./run_classification.sh "1+_25" "1+_25_rpr" "1+" "svm"
#./run_classification.sh "2+_1" "2+_1_rpr" "2+" "svm"
#./run_classification.sh "2+_8" "2+_8_rpr" "2+" "svm"
#./run_classification.sh "2+_9" "2+_9_rpr" "2+" "svm"


images=("1+_20" "1+_25" "2+_1" "2+_8" "2+_9" "3+_19")
type_im=("1+" "1+" "2+" "2+" "2+" "3+")
methods="stratified_random" #"halton_patches" # random_patches"
#images=("1+_20")
#type_im=("1+")
classifiers="knn" #knn deep svm"
for clf in $classifiers;do
for m in $methods;do
for i in `seq 0 4`;do
	./run_classification.sh ${images[$i]} "rpr" ${type_im[$i]} ${clf} 0 2 $m 2
done;
done;
done;


#./run_classification.sh "1+_20" "rpr" "1+" "knn" 1 2 "halton_patches" 2
#./run_classification.sh "1+_25" "rpr" "1+" "knn" 1 2 "halton_patches" 2
#./run_classification.sh "2+_1" "rpr" "2+" "knn" 1 2 "halton_patches" 2
##./run_classification.sh "2+_8" "2+_8_rpr" "2+" "deep" under_sampling cluster "halton_patches" local_rw
#./run_classification.sh "2+_8" "rpr" "2+" "knn" 1 2 "halton_patches" "2"
#./run_classification.sh "2+_9" "rpr" "2+" "knn" 1 2 "halton_patches" 2
#./run_classification.sh "3+_19" "rpr" "3+" "knn" 1 2 "halton_patches" 2


