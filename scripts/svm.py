# In[1]:
# Python libraries
import sys

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV



#from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score,  precision_score, f1_score, confusion_matrix)
from skimage import io
from matplotlib import pyplot as plt
import time


base_dir = "/home/rpezoa/experiment_data/"
img_name = sys.argv[2]
img_type = sys.argv[3]
seed = sys.argv[4]
images=[img_name]
type_im = [img_type]
feat = "rpr"
feat_path_big = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_" + feat + ".npy"
feat_path = base_dir + "output/" + images[0] + "_" + feat + "_halton_patches/feat_vectors/" + seed + ".npy"  
label_path_big = base_dir + "big_" + type_im[0] + "/labels/" + images[0] + ".npy"
label_path = base_dir + "output/" + images[0] + "_" + feat + "_halton_patches/target_vectors/" + seed + ".npy"  
blue_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_b.npy"
#dab_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_dab.npy"
print(feat_path, label_path)


# In[2]:
# Load features
img_path = base_dir +"big_" + type_im[0] + "/img/" + images[0] + ".tif"
img = io.imread(img_path)

big_X = np.load(feat_path_big)
big_y = np.load(label_path_big)
X = np.load(feat_path)
blue = np.load(blue_path)
#dab = np.load(dab_path)
n_feat = X.shape[1]
print("N feat:", n_feat)
y = np.load(label_path)
print(X.shape, y.shape)



# In[9]:

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


def print_metrics(s,target,prediction):
    cm = confusion_matrix(target, prediction)
    print("-"*10,s,"-"*10)
    print(cm)
    f1 = f1_score(target,prediction)
    rec = recall_score(target, prediction)
    prec = precision_score(target,prediction)
    print("f1", f1)
    print("rec", rec)
    print("prec", prec)
    #print("roc", roc_auc_score(target,prediction))
    return cm, f1, prec, rec



def run_svm(training_size,out_dir):
    sizes = 1
    F1 = np.zeros((sizes,3))
    PREC = np.zeros((sizes,3))
    REC = np.zeros((sizes,3))
    #TIME = np.zeros((sizes,1))

    #x_train, y_train,idx_0,idx_1 = training_data(X,y,size=training_size,p_class1=perc) #size=20000,p_class1=0.2)
    #m_idx0 = matrix_indices(idx_0)
    #m_idx1 = matrix_indices(idx_1)
    x_train_2, x_test, y_train_2, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    
    sfm = SelectFromModel(LassoCV())
    sfm.fit(x_train_2, y_train_2)
    train = sfm.transform(x_train_2)
    test = sfm.transform(x_test)
    XX = sfm.transform(big_X)

    
    kernel = 'rbf'
    C_r = [-3, 3]
    C_step = 0.5
    g_r = [-3, 3]
    g_step = 0.5
    C_range = 10. ** np.arange(C_r[0], C_r[1],C_step)
    gamma_range = 10. ** np.arange(g_r[0], g_r[1],g_step)
    # score can be: roc_auc, accuracy, recall, precision, f1, average_precision
    #score = 'average_precision'
    score="f1"


    pred_training, pred_big_im, pred_testing, best_params = svm(train,y_train_2,XX,test,kernel,gamma_range,C_range,score)
    #print(best_params)
    best_C = best_params["C"]
    best_gamma = best_params["gamma"]
    best_k = best_params["kernel"]
    
    #preffix = out_dir+"/"+str(ep)+"_"+str(bs)+"_"+ str(perc) + "_"
    preffix = out_dir+"/" + img_name + "/svm/" + "_".join([str(best_C),str(best_gamma),best_k,""])
    
    
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
    f1_train.write(str(trainSize)+":"+str(F1[i,0])+"\n") 
    prec_train.write(str(trainSize)+":"+str(PREC[i,0])+"\n") 
    rec_train.write(str(trainSize)+":"+str(REC[i,0])+"\n") 
    
    cm_test,F1[i,1],PREC[i,1],REC[i,1] = print_metrics("Test",y_test, pred_testing)
    f1_test.write(str(trainSize)+":"+str(F1[i,1])+"\n")
    prec_test.write(str(trainSize)+":"+str(PREC[i,1])+"\n") 
    rec_test.write(str(trainSize)+":"+str(REC[i,1])+"\n") 
    
    cm_big,F1[i,2],PREC[i,2],REC[i,2] = print_metrics("Big image",big_y,pred_big_im)
    f1_big.write(str(trainSize)+":"+str(F1[i,2])+"\n")
    prec_big.write(str(trainSize)+":"+str(PREC[i,2])+"\n") 
    rec_big.write(str(trainSize)+":"+str(REC[i,2])+"\n") 
    
    scores = str(np.round(F1[i,2],2))+"_" +str(np.round(PREC[i,2],2)) + "_" + str(np.round(REC[i,2],2))
    plt.imsave(preffix  + str(training_size) + "_"+  scores +"_pred.png",np.reshape(pred_big_im,(1000,1000)))
             
    f1_train.close()
    f1_test.close()
    f1_big.close()

outDir=sys.argv[1]
trainSize=X.shape[0]

print("---->",outDir)
run_svm(trainSize,outDir)
#run_ann(100,300,0.2,2000)
