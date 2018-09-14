img_name="3+_19"
img_type="3+"
outputDir="../notebooks/out"
test_perc=0.2
percentages=0.2 #0.3" #percentages of mem pixels
seeds="1 2 3 4 5 6 7 8 9" #0
trainSize=13005
for s in $seeds;do
	python svm_strati.py $outputDir ${img_name} $img_type $s $percentages $test_perc $trainSize
done;

