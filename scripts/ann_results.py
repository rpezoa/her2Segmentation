base_dir = "/home/rpezoa/her2Segmentation/notebooks/out/"

batchSize=["100", "300"]
epochs=["100", "200"]
percentages=["0.2", "0.3"]
trainingSize=["1000", "2000", "3000", "4000", "5000", "6000"]

#a2 = open("F1_test.txt")
#a3 = open("F1_big.txt")
d={}
#d2={}
#d3={}

for bs in batchSize:
    for e in epochs:
        for p in percentages:
        
            preffix = "_".join([str(bs),str(e),str(p),""])
            a1 = open(base_dir +preffix + "F1_train.txt")
            for l1 in a1:
                lista1=l1.strip().split(":")
                trainSize,f1 = lista1[0],lista1[1]
                if trainSize not in d:
                    d[trainSize] = []
                else:
                    d[trainSize].append((f1,bs,e,p))
            a1.close()
print(d)
print(len(d))
