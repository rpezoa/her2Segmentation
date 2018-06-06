images=("1+_20" "1+_25" "2+_1" "2+_8" "2+_9" "3+_19")
type_im=("1+" "1+" "2+" "2+" "2+" "3+")

for i in `seq 0 5`;do
        ./stratified_random.sh ${images[$i]} ${images[$i]}_rpr ${type_im[$i]}
done;

#./stratified_random.sh "1+_20" "1+_20_rpr" "1+"
