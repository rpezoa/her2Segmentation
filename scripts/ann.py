import sys

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (recall_score,  precision_score, f1_score, roc_auc_score,
                             make_scorer, confusion_matrix)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import class_weight


import random as rd
from skimage import io
from matplotlib import pyplot as plt
import time


base_dir = "/home/rpezoa/experiment_data/"
img_name = sys.argv[6]
img_type = sys.argv[7]
images=[img_name]
type_im = [img_type]
feat = "rpr"
feat_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_" + feat + ".npy"
label_path = base_dir + "big_" + type_im[0] + "/labels/" + images[0] + ".npy"
blue_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_b.npy"
#dab_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_dab.npy"
print(feat_path, label_path)



img_path = base_dir +"big_" + type_im[0] + "/img/" + images[0] + ".tif"
img = io.imread(img_path)


X = np.load(feat_path)
blue = np.load(blue_path)
#dab = np.load(dab_path)
n_feat = X.shape[1]
print("N feat:", n_feat)
y = np.load(label_path)
print(X.shape, y.shape)



def training_data(X,y, method="strati", size=13005, p_class1=0.3,seed=0):
    if method == "strati":
        mask_0 = y == 0
        mask_1 = y == 1
        idx_0 = np.where(mask_0 == True)[0] #indices of pixels with target equal 0 (non-mem) 
        idx_1 = np.where(mask_1 == True)[0] #indices of pixels with target equal 1 (mem)
        print("idx_0.shape", idx_0.shape, "idx_1.shape", idx_1.shape)
        n_0 = mask_0.sum()

        n_1 = mask_1.sum()
        print("non-mem:", n_0, "mem:", n_1)
        y_0 = y[mask_0]
        y_1 = y[mask_1]

        X_0 = X[mask_0] #only non-mem
        X_1 = X[mask_1] #only mem
        print("X_0.shape, X_1.shape", X_0.shape, X_1.shape)

        rd.seed(seed)
        rd_idx_0 = rd.sample(range(n_0),int((1-p_class1) * size))
        rd_idx_1 = rd.sample(range(n_1),int(p_class1 * size))


        X_0 = X_0[rd_idx_0,:]
        X_1 = X_1[rd_idx_1,:]

        y_0 = y_0[rd_idx_0]
        y_1 = y_1[rd_idx_1]

        new_idx_0 = idx_0[rd_idx_0]
        new_idx_1 = idx_1[rd_idx_1]

        rd_X = np.concatenate((X_0,X_1),axis=0)
        rd_y = np.concatenate((y_0,y_1),axis=0)
        print("X.shape:", rd_X.shape, "y.shape", rd_y.shape)
        return rd_X, rd_y,new_idx_0,new_idx_1
    


def matrix_indices(a,dim=(1000,1000)):
    idxs = np.unravel_index(a, dim)
    return idxs


def show_training_pixels(big_im,idxs1,idxs2):
    n = idxs1[0].shape[0]
    the_im2 = big_im[:,:,0:3].copy()
    for i in range(n):
        the_im2[idxs1[0][i],idxs1[1][i],0] = 0
        the_im2[idxs1[0][i],idxs1[1][i],1] = 0
        the_im2[idxs1[0][i],idxs1[1][i],2] = 255
        
    n2 = idxs2[0].shape[0]
    for i in range(n2):
        the_im2[idxs2[0][i],idxs2[1][i],0] = 255
        the_im2[idxs2[0][i],idxs2[1][i],1] = 0
        the_im2[idxs2[0][i],idxs2[1][i],2] = 0
    
    return the_im2


def f1_(y_true, y_pred):
    def recall_(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision_(y_true, y_pred)
    recall = recall_(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[9]:


def get_model():
    print("Input_dim", input_dim)
    classifier = Sequential()
    # Adding the input layer and the first hidden layer with dropout
    #print("initial dim:", initial_dim)
    classifier.add(Dense(activation = 'relu', input_dim = input_dim, units = 8, kernel_initializer = 'uniform'))
    #Randomly drops 0.1, 10% of the neurons in the layer.
    classifier.add(Dropout(rate= 0.1))

    #Adding the second hidden layer
    classifier.add(Dense(activation = 'relu', units = 8, kernel_initializer = 'uniform'))
    #Randomly drops 0.1, 10% of the neurons in the layer.
    classifier.add(Dropout(rate = 0.1))

    # Adding the output layer
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [f1_])
    
    return classifier

def ann(X,y,big_X,ep,batchS,y_train,test):
    t1 = time.time()
    clf  = KerasClassifier(build_fn=get_model)
    epochs  = [ep]
    batch_size = [batchS]
    shuffle = [False]
    class_weights = [class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)]
    print(class_weights,X.shape[1])
    param_grid = dict(epochs=epochs, batch_size=batch_size, class_weight=class_weights,shuffle=shuffle)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1,cv=10, scoring="f1", verbose=0)
    # Fitting the ANN to the training set
    #classifier.fit(X, y, batch_size = 200, epochs = 300, class_weight =class_weights, 
                   #shuffle=False,verbose=1)
    
    grid_result = grid.fit(X,y)
  
    pred_big_im = grid_result.best_estimator_.predict(big_X)
    pred_training = grid_result.best_estimator_.predict(X)
    pred_testing = grid_result.best_estimator_.predict(test)
    #clf.fit(x_train, y_train, batch_size = 10, nb_epoch = 1000)
    #pred_big_im = clf.predict(big_scaled_feat_matrix)
    pred_big_im = (pred_big_im > 0.5)
    pred_training = (pred_training > 0.5)
    pred_testing = (pred_testing > 0.5)
    t2 = time.time()
    t = t2 - t1
    print("Time:::::::", t)
    print("Best: %f using %s"% (grid_result.best_score_, grid_result.best_params_))
    return pred_training, pred_big_im, pred_testing


def print_metrics(s,target,prediction):
    cm = confusion_matrix(target, prediction)
    print("-"*10,s,"-"*10)
    print(cm)
    f1 = f1_score(target,prediction)
    rec = recall_score(target, prediction)
    prec = precision_score(target,prediction)
    print("f1", f1)
    print("rec", prec)
    print("prec", rec)
    #print("roc", roc_auc_score(target,prediction))
    return cm, f1,prec,rec

def measure_bronw(c,t='auto'):
    plt.hist(c, bins=t)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

def run_ann(ep,bs,perc,training_size,out_dir):
    sizes = 1
    F1 = np.zeros((sizes,3))
    PREC = np.zeros((sizes,3))
    REC = np.zeros((sizes,3))
    #TIME = np.zeros((sizes,1))

    x_train, y_train,idx_0,idx_1 = training_data(X,y,size=training_size,p_class1=perc) #size=20000,p_class1=0.2)
    #m_idx0 = matrix_indices(idx_0)
    #m_idx1 = matrix_indices(idx_1)
    x_train_2, x_test, y_train_2, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    sfm = SelectFromModel(LassoCV())
    sfm.fit(x_train_2, y_train_2)
    train = sfm.transform(x_train_2)
    test = sfm.transform(x_test)
    XX = sfm.transform(X)

    global input_dim 
    input_dim = train.shape[1]
    i=0
    #pred_training, pred_big_im, pred_testing = ann(x_train_2,y_train_2,X)
    input_dim = train.shape[1]
    pred_training, pred_big_im, pred_testing = ann(train,y_train_2,XX,ep,bs,y_train,test)
    
    #preffix = out_dir+"/"+str(ep)+"_"+str(bs)+"_"+ str(perc) + "_"
    preffix = out_dir+"/" + img_name + "/" + "_".join([str(ep),str(bs),str(perc),""])
    
    
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
    
    
    
    cm_train,F1[i,0],PREC[i,0],REC[i,0] = print_metrics("Trainng",y_train_2,pred_training)
    f1_train.write(str(trainSize)+":"+str(F1[i,0])+"\n") 
    prec_train.write(str(trainSize)+":"+str(PREC[i,0])+"\n") 
    rec_train.write(str(trainSize)+":"+str(REC[i,0])+"\n") 
    
    cm_test,F1[i,1],PREC[i,1],REC[i,1] = print_metrics("Test",y_test, pred_testing)
    f1_test.write(str(trainSize)+":"+str(F1[i,1])+"\n")
    prec_test.write(str(trainSize)+":"+str(PREC[i,1])+"\n") 
    rec_test.write(str(trainSize)+":"+str(REC[i,1])+"\n") 
    
    cm_big,F1[i,2],PREC[i,2],REC[i,2] = print_metrics("Big image",y,pred_big_im)
    f1_big.write(str(trainSize)+":"+str(F1[i,2])+"\n")
    prec_big.write(str(trainSize)+":"+str(PREC[i,2])+"\n") 
    rec_big.write(str(trainSize)+":"+str(REC[i,2])+"\n") 
    
    scores = str(np.round(F1[i,2],2))+"_" +str(np.round(PREC[i,2],2)) + "_" + str(np.round(REC[i,2],2))
    plt.imsave(preffix  + str(training_size) + "_"+  scores +"_pred.png",np.reshape(pred_big_im,(1000,1000)))
             
    f1_train.close()
    f1_test.close()
    f1_big.close()


epochs=int(sys.argv[1])
batchSize=int(sys.argv[2])
percentage=float(sys.argv[3])
print("<<<<<<<<<", sys.argv[4])
trainSize=int(sys.argv[4])
outDir=sys.argv[5]

print("---->",epochs,batchSize,percentage,trainSize,outDir)
run_ann(epochs,batchSize,percentage,trainSize,outDir)
#run_ann(100,300,0.2,2000)
