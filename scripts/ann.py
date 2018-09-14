import sys

import numpy as np
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import GridSearchCV
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import class_weight

from skimage import io
from matplotlib import pyplot as plt
import time
from common import strati_training_data, print_metrics, feat_sel


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

def ann(X,y,big_X,ep,batchS,test):
    ##ann(x_train_2, y_train_2, XX, ep, bs, y_train,test)
    
    t1 = time.time()
    clf  = KerasClassifier(build_fn=get_model)
    epochs  = [ep]
    batch_size = [batchS]
    shuffle = [False]
    class_weights = [class_weight.compute_class_weight('balanced', np.unique(y),y)]
    print(class_weights,X.shape[1])
    param_grid = dict(epochs=epochs, batch_size=batch_size, class_weight=class_weights,shuffle=shuffle)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1,cv=10, scoring="f1", verbose=1)
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



def run_ann(X,y,ep,bs,perc,training_size,seed,out_dir):
    sizes = 1
    F1 = np.zeros((sizes,3))
    PREC = np.zeros((sizes,3))
    REC = np.zeros((sizes,3))
    
    # Stratified random data
    x_train, y_train, idx_0, idx_1 = strati_training_data(X,y,size=training_size,
                                                       p_class1=perc,
                                                       seed=int(seed)) #size=20000,p_class1=0.2)
    
    print("x_train, y_train", x_train.shape, y_train.shape)
    
    
    rus = RandomUnderSampler(random_state=0)
    x_res, y_res = rus.fit_sample(x_train, y_train)
    
    #t1=time.time()
    #sfm = feat_sel(x_res, y_res)
    #x_sel = sfm.transform(x_res)
    #t2=time.time()
    x_sel = x_res
    #print("Feature selection time:", t2-t1)
    
    x_train_2, x_test, y_train_2, y_test = train_test_split(x_sel, y_res, 
                                                            test_size=test_perc, 
                                                            random_state=0)
    #sfm = SelectFromModel(LassoCV())
    #sfm.fit(x_train_2, y_train_2)
    #train = sfm.transform(x_train_2)
    #XX = sfm.transform(X)
    XX=X
    global input_dim 
    input_dim = x_train_2.shape[1]
    
    #pred_training, pred_big_im, pred_testing = ann(x_train_2,y_train_2,X)
    pred_training, pred_big_im, pred_testing = ann(x_train_2, y_train_2, XX, ep, bs,x_test)
    
    #preffix = out_dir+"/"+str(ep)+"_"+str(bs)+"_"+ str(perc) + "_"
    preffix = out_dir+"/" + img_name + "/ann_strati_0/" + "_".join([str(ep),str(bs),str(perc),seed,""])
    
    
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
    prec_train.write(str(trainSize)+":"+str(seed)+":"+str(PREC[i,0])+"\n") 
    rec_train.write(str(trainSize)+":"+str(seed)+":"+str(REC[i,0])+"\n") 
    
    cm_test,F1[i,1],PREC[i,1],REC[i,1] = print_metrics("Test",y_test, pred_testing)
    f1_test.write(str(trainSize)+":"+str(seed)+":"+str(F1[i,1])+"\n")
    prec_test.write(str(trainSize)+":"+str(seed)+":"+str(PREC[i,1])+"\n") 
    rec_test.write(str(trainSize)+":"+str(seed)+":"+str(REC[i,1])+"\n") 
    
    cm_big,F1[i,2],PREC[i,2],REC[i,2] = print_metrics("Big image",y,pred_big_im)
    f1_big.write(str(trainSize)+":"+str(seed)+":"+str(F1[i,2])+"\n")
    prec_big.write(str(trainSize)+":"+str(seed)+":"+str(PREC[i,2])+"\n") 
    rec_big.write(str(trainSize)+":"+str(seed)+":"+str(REC[i,2])+"\n") 
    
    scores = str(np.round(F1[i,2],2))+"_" +str(np.round(PREC[i,2],2)) + "_" + str(np.round(REC[i,2],2))
    plt.imsave(preffix  + str(training_size) + "_"+  scores +"_pred.png",np.reshape(pred_big_im,(1000,1000)))
             
    f1_train.close()
    f1_test.close()
    f1_big.close()


batchSize = int(sys.argv[1])
epochs = int(sys.argv[2])
percentage = float(sys.argv[3])
trainSize = int(sys.argv[4])
outDir = sys.argv[5]
img_name = sys.argv[6]
img_type = sys.argv[7]
seed = sys.argv[8]
test_perc = float(sys.argv[9])
print("---->",epochs,batchSize,percentage,trainSize,outDir)

base_dir = "/home/rpezoa/experiment_data/"
images=[img_name]
type_im = [img_type]
feat = "rpr"
feat_path = base_dir + "big_" + type_im[0] +"/features/" + images[0] + "_" + feat + ".npy"
label_path = base_dir + "big_" + type_im[0] + "/labels/" + images[0] + ".npy"
print(feat_path, label_path)


img_path = base_dir +"big_" + type_im[0] + "/img/" + images[0] + ".tif"
img = io.imread(img_path)

X = np.load(feat_path)
n_feat = X.shape[1]
print("N feat:", n_feat)
y = np.load(label_path)
print(X.shape, y.shape)

run_ann(X,y,epochs,batchSize,percentage,trainSize,seed,outDir)
