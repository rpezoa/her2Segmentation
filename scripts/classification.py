import sys
import os
sys.path.append('/home/rpezoa/svm/')
#import warnings
#warnings.filterwarnings('ignore')
import numpy as np
import time
from data import under_sampling as under_s
import argparse
from shutil import copyfile
from sklearn.externals import joblib

import random as rd
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rd.seed(12345)

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (recall_score,  precision_score, f1_score, roc_auc_score, 
                             make_scorer, confusion_matrix)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from config_A import *

################################################################################
# Parsing data 
#   required parameters: input_path output_path image
#   optional parameters: features

parser = argparse.ArgumentParser(description='Testing images.')
parser.add_argument('input_path', help='Directory of features')
parser.add_argument('output_path', help='Directory to save results' )
parser.add_argument('image',help='Image name without extension')
parser.add_argument('-score','--score', help="SVM Performance Score")
parser.add_argument('-btp','--big_target_path', help="Big Target Path")
parser.add_argument('-is','--image_size', help="Image size (square image)", type=int)
parser.add_argument('-us','--under_sampling', help="Undersampling", type=int)
parser.add_argument('-tp','--target_path', help="Target Path")
parser.add_argument('-bp', '--big_path', help="Big scaled feature Path")
parser.add_argument('-feat_path', '--features_path', help="Feature Path")
parser.add_argument('-cluster', '--cluster', help="Cluster flag", type=int)
parser.add_argument('-seed','--seed', help="Seed", type=int)
parser.add_argument('-clf','--classifier', help="Classifier")
parser.add_argument('-lrw','--local_rw', help="Local read-write", type=int)
args = parser.parse_args()

args.input_path = args.input_path + '/' if args.input_path[-1] != '/' else args.input_path
args.output_path = args.output_path + '/' if args.output_path[-1] != '/' else args.output_path

seed = int(args.seed)
under_sampling = int(args.under_sampling)
if args.image_size:
    im_size = args.image_size
if args.features_path:
    feat_path = args.features_path
if args.target_path:
    target_path = args.target_path
if args.big_path:
    big_path = args.big_path
if args.big_target_path:
    big_target_path = args.big_target_path
if args.classifier:
    c = args.classifier
if args.cluster:
    cluster = args.cluster
if args.local_rw:
    local_rw = args.local_rw
print ("Directory of Data: %s" % args.input_path )
print ("Output Directory: %s" % args.output_path)
print ("Image to Process: %s" % args.image )

baseDir = args.input_path
imgDir =  baseDir + 'img/'
maskDir = baseDir + 'masks/'
featDir = baseDir + 'features/'
out_dir = args.output_path 

image_name = args.image.split('.')[0]
print ("Feature Set Directory: %s" % args.input_path )
im_name =  args.image

def get_data(local_path, ext_path):
    if not os.path.exists(local_path + ext_path):
        print("Copying data to tmp", ext_path, "::::::",local_path + ext_path)
        new_dir = local_path + ext_path
        new_dir = new_dir.split("/")[0:-1]
        new_dir = "/".join(new_dir)
        os.makedirs(new_dir)
        copyfile(ext_path, local_path + ext_path)
    data = np.load(local_path + ext_path)
    return data
        
print("\n" + "%"*20 +" Seed: " +str(seed) + "%"*20 )
local_path = "/tmp/"
if local_rw == 1: #reading/writing local in /tmp
    big_scaled_feat_matrix = get_data(local_path,big_path)
    big_target_vector = get_data(local_path,big_target_path)
    current_X = get_data(local_path,feat_path)
    current_y = get_data(local_path,target_path)
else:
    big_scaled_feat_matrix = np.load(big_path)
    big_target_vector = np.load(big_target_path)
    current_X = np.load(feat_path)
    current_y = np.load(target_path)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_dir + "/big_pred"):
    os.makedirs(out_dir + "big_pred")
if not os.path.exists(out_dir + "/pixels_pred"):
    os.makedirs(out_dir + "pixels_pred")
if not os.path.exists(out_dir + "/times"):
    os.makedirs(out_dir + "times")
if not os.path.exists(out_dir + "/cm"):
    os.makedirs(out_dir + "cm")
if not os.path.exists(out_dir + "/cm_train"):
    os.makedirs(out_dir + "cm_train")
if not os.path.exists(out_dir + "/cm_test"):
    os.makedirs(out_dir + "cm_test")
if not os.path.exists(out_dir + "/big_prob"):
    os.makedirs(out_dir + "big_prob")
if not os.path.exists(out_dir + "/test_pred"):
    os.makedirs(out_dir + "test_pred")
if not os.path.exists(out_dir + "/params"):
    os.makedirs(out_dir + "params")

nX,ny= current_y.sum(), current_y.sum()
print(ny)


if under_sampling == 1:
# Here, it is the undersampling
    current_X, current_y = under_s(current_X, current_y,ny,ny )
    print("Features vectors undersampled")
n_mem_train = current_y.sum()
print("current_X.shape",current_X.shape)
print("curret_y.shape", current_y.shape)
c_n_mem_pixels = (current_y==1).sum()

print('Current mem pixels: ', c_n_mem_pixels)
print('Current non-mem pixels:', (current_y==0).sum())
print("N features:", current_X.shape[1])
print("current_X.mean()", current_X.mean())

N = current_X.shape[0]

def save_results(best_score, best_params,t,clf):
    joblib.dump(best_score, out_dir  + "params/" + str(seed) + "_score.pkl")
    joblib.dump(best_params, out_dir  + "params/" + str(seed) + "_params.pkl")
    np.save(out_dir + "times/" + str(seed) + ".npy", t)
    print("Best: %f using %s"% (best_score, best_params))
    print(clf, "time:", t)

def svm_c(X,y,big_X,x_test=None):
    print("SVM :::::: X,y", X.shape, y.shape) 
    t1=time.time()
    tuned_parameters = [{'kernel': [kernel], 'gamma': gamma_range,'C': C_range}]
    clf = GridSearchCV(SVC(class_weight="balanced"), tuned_parameters, cv=k, scoring=score,n_jobs = -1, verbose=0)
    clf.fit(x_train, y_train)
    best_estimator = clf.best_estimator_
    pred_vector = best_estimator.predict(X)
    pred_big_im = best_estimator.predict(big_X)
    pred_test = best_estimator.predict(x_test)
    t2=time.time()
    t = t2 - t1
    save_results(clf.best_score_, clf.best_params_,t,"svm")
    return pred_vector, pred_big_im, pred_test

from keras import backend as K

from sklearn.metrics import average_precision_score

def aver_prec(y_true,y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    return average_precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def create_model():
    activation="relu"
    init_mode = "uniform"
    model = Sequential()
    model.add(Dense(units = 8,  kernel_initializer="uniform", activation="relu", input_dim=40))
    model.add(Dense(units = 8,  kernel_initializer="uniform", activation="relu", input_dim=8))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [precision])
    return model
    
def deep_l(X,y,big_X):

    print("Deep Learning :::::: X,y", X.shape, y.shape) 
    #clf = Sequential()
    #clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 40))
    #clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    #clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    ## https://datascience.stackexchange.com/questions/13746/how-to-define-a-custom-performance-metric-in-keras
    #clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    t1 = time.time()
    model = KerasClassifier(build_fn=create_model)
    
    epochs = [1000]
    batch_size = [350]
    param_grid = dict(epochs=epochs, batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2, verbose=2,scoring="precision")
    grid_result = grid.fit(x_train, y_train)
    pred_big_im= grid_result.best_estimator_.predict(big_X)
    pred_big_im = (pred_big_im > 0.5)
    #clf.fit(x_train, y_train, batch_size = 10, nb_epoch = 1000)
    #pred_big_im = clf.predict(big_scaled_feat_matrix)
    #pred_big_im = (pred_big_im > 0.5)
    t2 = time.time()
    t = t2 - t1
    save_results(grid_result.best_score_, grid_result.best_params_,t,"Deep")
    return pred_big_im, pred_big_im, pred_big_im

def ann(X,y,big_X):
    t1 = time.time()
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)
    print("__________________------------------",class_weights)

    classifier = Sequential()
    # Adding the input layer and the first hidden layer with dropout
    classifier.add(Dense(activation = 'relu', input_dim = 40, units = 8, kernel_initializer = 'uniform'))
    #Randomly drops 0.1, 10% of the neurons in the layer.
    classifier.add(Dropout(rate= 0.1))

    #Adding the second hidden layer
    classifier.add(Dense(activation = 'relu', units = 8, kernel_initializer = 'uniform'))
    #Randomly drops 0.1, 10% of the neurons in the layer.
    classifier.add(Dropout(rate = 0.1)) 

    # Adding the output layer
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [f1])

    # Fitting the ANN to the training set
    classifier.fit(x_train, y_train, batch_size = 200, epochs = 100, class_weight =class_weights, shuffle=False,verbose=0)

    pred_big_im = classifier.predict(big_X)
    pred_training = classifier.predict(x_train)
    pred_testing = classifier.predict(x_test)
    #clf.fit(x_train, y_train, batch_size = 10, nb_epoch = 1000)
    #pred_big_im = clf.predict(big_scaled_feat_matrix)
    pred_big_im = (pred_big_im > 0.5)
    pred_training = (pred_training > 0.5)
    pred_testing = (pred_testing > 0.5)
    t2 = time.time()
    t = t2 - t1
    print("Time:::::::", t)
    return pred_training, pred_big_im, pred_testing


def random_f(X,y,big_X):
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    print("Random forest :::::: X,y", X.shape, y.shape)
    t1=time.time()
    param_grid = {'criterion': ['gini', 'entropy'],
    'max_features': [2, 3,4],
    'min_samples_leaf': [2,3,4],
    'n_estimators': [5, 15, 25,35]}
    clf = RandomForestClassifier(random_state=0)
    grid_result = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 10)
    grid_result.fit(x_train, y_train)
    pred_vector = grid_result.best_estimator_.predict(X)
    pred_big_im = grid_result.best_estimator_.predict(big_X)
    t2=time.time()
    t = t2-t1
    save_results(grid_result.best_score_, grid_result.best_params_,t,"RF")
    return pred_vector, pred_big_im, pred_big_im

def kNN(X,y,big_X):
    print("KNN :::::: X,y", X.shape, y.shape) 
    t1 = time.time()
    param_grid = {'n_neighbors': np.arange(20)+1, 'weights': ['uniform', 'distance']}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn,param_grid,cv=5, scoring="f1", n_jobs = -1, verbose=2)
    print("::::::: Training with ", x_train.shape, ":::::::::::::")
    grid_result = clf.fit(x_train, y_train)
    pred_vector = grid_result.best_estimator_.predict(X)
    pred_big_im = grid_result.best_estimator_.predict(big_X)
    t2 = time.time()
    t = t2 - t1
    save_results(grid_result.best_score_, grid_result.best_params_,t,"KNN")
    return pred_vector, pred_big_im, pred_big_im


def extra_trees(X,y,big_X):
    print("Extra Trees :::::: X,y", X.shape, y.shape)
    t1 = time.time()
    param_grid = {'n_estimators':[5,15,25,35],'class_weight':['balanced'], 'criterion':["gini","entropy"]} 
    et = ExtraTreesClassifier(random_state=0)
    clf = GridSearchCV(estimator = et, param_grid=param_grid,cv=5, n_jobs=-1, verbose=2) 
    print("::::::: Training with ", x_train.shape, ":::::::::::::")
    grid_result = clf.fit(x_train, y_train)
    pred_vector = grid_result.best_estimator_.predict(X)
    pred_big_im = grid_result.best_estimator_.predict(big_X)
    t2=time.time()
    t = t2 - t1
    save_results(grid_result.best_score_, grid_result.best_params_,t,"ET")
    return pred_vector, pred_big_im, pred_big_im

def xgboost(X,y,big_X):
    # https://www.kdnuggets.com/2017/03/simple-xgboost-tutorial-iris-dataset.html
    import xgboost as xgb
    print("XGBoost :::::: X,y", X.shape, y.shape) 
    dtrain = xgb.DMatrix(X, label=y)
    print(y)
    # specify parameters via map
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic', 'num_class':2 }
    num_round = 5
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    #------------------------------------------------------  
    # Generating predictions and saving the predictions
    pred_big_im = bst.predict(big_X)
    return pred_big_im, pred_big_im,pred_big_im

def print_metrics(s,target,prediction):
    cm = confusion_matrix(target, prediction)
    print("-"*10,s,"-"*10)
    print(cm)
    print("f1", f1_score(target,prediction))
    print("rec", recall_score(target, prediction))
    print("prec", precision_score(target,prediction))
    print("roc", roc_auc_score(target,prediction))
    return cm
    

if c_n_mem_pixels > 3:
    x_train, x_test, y_train, y_test = train_test_split(current_X, current_y, test_size=0.6, random_state=0)
    print("*"*5, x_train.mean(), x_test.mean(), y_train.sum(), y_test.sum())
    if c == "svm":
        pred_vector, pred_big_im, pred_t = svm_c(x_train,y_train, big_scaled_feat_matrix, x_test)
    elif c == "deep":
        pred_vector, pred_big_im, pred_t  = deep_l(x_train, y_train, big_scaled_feat_matrix)
    elif c == "rf":
        pred_vector, pred_big_im, pred_t = random_f(x_train,y_train, big_scaled_feat_matrix)
    elif c == "knn":
        pred_vector, pred_big_im, pred_t = kNN(x_train,y_train, big_scaled_feat_matrix)
    elif c == "xgboost":
        pred_vector, pred_big_im, pred_t = xgboost(x_train, y_train, big_scaled_feat_matrix)
    elif c == "extra_trees":
        pred_vector, pred_big_im, pred_t = extra_trees(x_train,y_train, big_scaled_feat_matrix)
    elif c == "ann":
        pred_vector, pred_big_im, pred_t  = ann(x_train, y_train, big_scaled_feat_matrix)

    
    # Generating predictions and saving the predictions
    print("Saving files in: ", out_dir)
    np.save(out_dir + "n_mem_train_seed_" + str(seed) + "_" + str(n_mem_train), n_mem_train)
    np.save(out_dir+"big_pred/"+ str(seed) +"_big_pred.npy", pred_big_im)
    np.save(out_dir+"pixels_pred/"+str(seed) +"_pixels_pred.npy", pred_vector)
    np.save(out_dir+"test_pred/"+ str(seed) +"_test_pred.npy", y_test)


    
    cm_big = print_metrics("Big Image",big_target_vector, pred_big_im)
    np.save(out_dir + "cm/" +  str(seed) + "_cm.npy", cm_big)
    cm_train = print_metrics("Training",y_train,pred_vector)
    np.save(out_dir + "cm_train/" +  str(seed) + "_cm.npy", cm_train)
    cm_test = print_metrics("Testing",y_test,pred_t)
    np.save(out_dir + "cm_test/" +  str(seed) + "_cm.npy", cm_test)


    if cluster ==0:
        from matplotlib import pyplot as plt
        plt.imsave(out_dir + "big_pred_seed_"+ str(seed) + ".png", np.reshape(pred_big_im, (im_size,im_size)), cmap="gray")
    else:
        print("Predictions are NOT saved as images, running on a cluster")

else:
    print("Not enough membrane pixels for training:", c_n_mem_pixels)
