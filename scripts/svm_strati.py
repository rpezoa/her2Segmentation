# In[1]:
# Python libraries
import sys

import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
from common import strati_training_data, print_metrics, feat_sel

def svm(X,y,big_X,x_test,kernel,gamma_range,C_range,score):
    print("X.shape",X.shape, "y.shape", y.shape)
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    t1=time.time()
    tuned_parameters = [{'kernel': [kernel], 'gamma': gamma_range,'C': C_range}]
    grid = GridSearchCV(SVC(class_weight="balanced"), tuned_parameters, cv=StratifiedKFold(n_splits=10), 
                        scoring=score, n_jobs = -1, verbose=1)
    grid.fit(X,y)    
    t2=time.time()
    t = t2 - t1
    print("Training time:::::::", t)
    print(grid)
    
    t1 = time.time()    
    pred_big_im = grid.best_estimator_.predict(big_X)
    t2 = time.time()
    t = t2 - t1
    print("Prediction time, big im. :::::::", t)

    t1 = time.time()        
    pred_training = grid.best_estimator_.predict(X)
    t2 = time.time()
    t = t2 - t1
    print("Prediction time,  training :::::::", t)

    t1 = time.time()    
    pred_testing = grid.best_estimator_.predict(x_test)
    t2 = time.time()
    t = t2 - t1
    print("Prediction time, testing :::::::", t)
   
    print("Best score: %f using %s"% (grid.best_score_, grid.best_params_))
    print("n_support:", grid.best_estimator_.n_support_)
    return pred_training, pred_big_im, pred_testing, grid.best_params_



def run_svm(training_size,out_dir):
    sizes = 1
    F1 = np.zeros((sizes,3))
    PREC = np.zeros((sizes,3))
    REC = np.zeros((sizes,3))
    #TIME = np.zeros((sizes,1))

    x_train, y_train,idx_0,idx_1 = strati_training_data(X,y,size=training_size,p_class1=perc,
                                                        seed=int(seed))
    rus = RandomUnderSampler(random_state=0)
    x_res, y_res = rus.fit_sample(x_train, y_train)
    
    #t1=time.time()
    #sfm = feat_sel(x_res, y_res)
    #x_sel = sfm.transform(x_res)
    #t2=time.time()
    #print("Feature selection time:", t2-t1)
    x_sel = x_res
                                                        
    x_train_2, x_test, y_train_2, y_test = train_test_split(x_sel, y_res, 
                                                            test_size=test_perc, 
                                                            random_state=0)
    #XX = sfm.transform(X)
    XX = X
    
    kernel = 'rbf'
    C_r = [-5, 5]
    C_step = 0.5
    g_r = [-5, 5]
    g_step = 0.5
    C_range = 10. ** np.arange(C_r[0], C_r[1],C_step)
    gamma_range = 10. ** np.arange(g_r[0], g_r[1],g_step)
    # score can be: roc_auc, accuracy, recall, precision, f1, average_precision
    #score = 'average_precision'
    score="f1"

    print(":::run_svm::: x_train_2.shape:", x_train_2.shape)
    pred_training, pred_big_im, pred_testing, best_params = svm(x_train_2,
                                                                y_train_2,XX,
                                                                x_test,kernel,
                                                                gamma_range,
                                                                C_range,score)
    #print(best_params)
    best_C = best_params["C"]
    best_gamma = best_params["gamma"]
    best_k = best_params["kernel"]
    
    #preffix = out_dir+"/"+str(ep)+"_"+str(bs)+"_"+ str(perc) + "_"
    preffix = out_dir+"/" + img_name + "/svm_strati_0/" + "_".join([str(best_C),str(best_gamma),best_k,seed,""])
    
    
    print("preffix", preffix)
    f1_train = open(preffix +"F1_train.txt","a")
    f1_test = open(preffix +"F1_test.txt","a")
    f1_big = open(preffix +"F1_big.txt","a")
    
    prec_train = open(preffix +"PREC_train.txt","a")
    prec_test = open(preffix +"PREC_test.txt","a")
    prec_big = open(preffix +"PREC_big.txt","a")
    
    rec_train = open(preffix +"REC_train.txt","a")
    rec_test = open(preffix +"REC_test.txt","a")
    rec_big = open(preffix +"REC_big.txt","a")
    
    i=0
    
    cm_train,F1[i,0],PREC[i,0],REC[i,0] = print_metrics("Trainng",y_train_2,pred_training)
    f1_train.write(str(trainSize)+":" +str(seed)+":"+str(F1[i,0])+"\n") 
    prec_train.write(str(trainSize)+":" +str(seed)+":"+str(PREC[i,0])+"\n") 
    rec_train.write(str(trainSize)+":" +str(seed)+":"+str(REC[i,0])+"\n") 
    
    cm_test,F1[i,1],PREC[i,1],REC[i,1] = print_metrics("Test",y_test, pred_testing)
    f1_test.write(str(trainSize)+":" +str(seed)+":"+str(F1[i,1])+"\n")
    prec_test.write(str(trainSize)+":" +str(seed)+":"+str(PREC[i,1])+"\n") 
    rec_test.write(str(trainSize)+":" +str(seed)+":"+str(REC[i,1])+"\n") 
    
    cm_big,F1[i,2],PREC[i,2],REC[i,2] = print_metrics("Big image",y,pred_big_im)
    f1_big.write(str(trainSize)+":" +str(seed)+":"+str(F1[i,2])+"\n")
    prec_big.write(str(trainSize)+":" +str(seed)+":"+str(PREC[i,2])+"\n") 
    rec_big.write(str(trainSize)+":" +str(seed)+":"+str(REC[i,2])+"\n") 
    
    scores = str(np.round(F1[i,2],2))+"_" +str(np.round(PREC[i,2],2)) + "_" + str(np.round(REC[i,2],2))
    plt.imsave(preffix  + str(training_size) + "_"+  scores +"_pred.png",np.reshape(pred_big_im,(1000,1000)))
             
    f1_train.close()
    f1_test.close()
    f1_big.close()


outDir=sys.argv[1]
img_name = sys.argv[2]
img_type = sys.argv[3]
seed = sys.argv[4]
perc = float(sys.argv[5])
test_perc = float(sys.argv[6])
trainSize = int(sys.argv[7])


base_dir = "/home/rpezoa/experiment_data/"
images=[img_name]
type_im = [img_type]
feat = "rpr"
feat_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_" + feat + ".npy"
label_path = base_dir + "big_" + type_im[0] + "/labels/" + images[0] + ".npy"
blue_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_b.npy"
#dab_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_dab.npy"
print(feat_path, label_path)



img_path = base_dir +"big_" + type_im[0] + "/img/" + images[0] + ".tif"

X = np.load(feat_path)

blue = np.load(blue_path)
#dab = np.load(dab_path)
n_feat = X.shape[1]
print("N feat:", n_feat)
y = np.load(label_path)
print(X.shape, y.shape)



print("---->",outDir)
run_svm(trainSize,outDir)
#run_ann(100,300,0.2,2000)
