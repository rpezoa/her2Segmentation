#python ann.py 100 300 0.2 1000
img_name="3+_19"
img_type="3+"
outputDir="../notebooks/out"
batchSize="100 300"
epochs="100 200 500"
percentages="0.2" #0.3" #percentages of mem pixels
trainingSize="13005" #"1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000"
seeds="0 1 2 3 4 5 6 7 8 9"
test_perc=0.2
for b in $batchSize;do
	for e in $epochs;do
		for p in $percentages;do
			for t in $trainingSize;do
				for s in $seeds;do
					echo $b $e $p $t $outputDir $s
					python ann.py $b $e $p $t $outputDir $img_name $img_type $s $test_perc
				done;
			done;
		done;
	done;
done;

