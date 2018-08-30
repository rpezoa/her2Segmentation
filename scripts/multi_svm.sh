#python ann.py 100 300 0.2 1000
img_name="1+_25"
img_type="1+"
outputDir="../notebooks/out"
seed=0
python svm.py $outputDir ${img_name} $img_type $seed

