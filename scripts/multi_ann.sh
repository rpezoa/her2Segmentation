#python ann.py 100 300 0.2 1000
img_name="1+_25"
img_type="1+"
outputDir="../notebooks/out"
batchSize="100 300"
epochs="100 200"
percentages="0.2 0.3"
trainingSize="1000 2000 3000 4000 5000 6000"
for b in $batchSize;do
	for e in $epochs;do
		for p in $percentages;do
			for t in $trainingSize;do
				echo $b $e $p $t $outputDir
				python ann.py $b $e $p $t $outputDir $img_name $img_type
			done;
		done;
	done;
done;

