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
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (recall_score,  precision_score, f1_score, roc_auc_score, 
                             make_scorer, confusion_matrix)
import keras
from keras.models import Sequential
from keras.layers import Dense
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
if not os.path.exists(out_dir + "/big_prob"):
    os.makedirs(out_dir + "big_prob")
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

def svm_c(X,y,big_X):
    print("SVM :::::: X,y", X.shape, y.shape) 
    tuned_parameters = [{'kernel': [kernel], 'gamma': gamma_range,'C': C_range}]
    clf = GridSearchCV(SVC(class_weight="balanced", probability=True), tuned_parameters, cv=k, scoring=score,n_jobs = -1, verbose=1)
    t1=time.time()
    clf.fit(x_train, y_train)
    t2=time.time()
    print("SVM fitting time:", t2-t1)
    best_estimator = clf.best_estimator_
    pred_vector = best_estimator.predict(X)
    pred_big_im = best_estimator.predict(big_X)
    prob_big_im = best_estimator.predict_proba(big_X)
    return pred_vector, pred_big_im, prob_big_im

def create_model():
    activation="relu"
    init_mode = "uniform"
    model = Sequential()
    model.add(Dense(units = 8,  kernel_initializer="uniform", activation="relu", input_dim=40))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
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
    
    epochs = [10,100]
    batch_size = [10, 50, 100]
    param_grid = dict(epochs=epochs, batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
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

def random_f(X,y,big_X):
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    print("Random forest :::::: X,y", X.shape, y.shape)
    t1=time.time()
    param_grid = {'criterion': ['gini', 'entropy'],
    'max_features': [2, 3,4],
    'min_samples_leaf': [2,3,4],
    'n_estimators': [5, 15, 25,35]}
    clf = RandomForestClassifier(random_state=0)
    grid_result = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
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

if c_n_mem_pixels > 3:
    x_train, x_test, y_train, y_test = train_test_split(current_X, current_y, test_size=0.2, random_state=0)
    if c == "svm":
        pred_vector, pred_big_im, prob_big_im = svm_c(x_train,y_train, big_scaled_feat_matrix)
    elif c == "deep":
        pred_vector, pred_big_im, prob_big_im  = deep_l(x_train, y_train, big_scaled_feat_matrix)
    elif c == "rf":
        pred_vector, pred_big_im, prob_big_im = random_f(x_train,y_train, big_scaled_feat_matrix)
    elif c == "knn":
        pred_vector, pred_big_im, prob_big_im = kNN(x_train,y_train, big_scaled_feat_matrix)
    elif c == "xgboost":
        pred_vector, pred_big_im, prob_big_im = xgboost(x_train, y_train, big_scaled_feat_matrix)
    elif c == "extra_trees":
        pred_vector, pred_big_im, prob_big_im = extra_trees(x_train,y_train, big_scaled_feat_matrix)

    
    # Generating predictions and saving the predictions
    print("Saving files in: ", out_dir)
    np.save(out_dir + "n_mem_train_seed_" + str(seed) + "_" + str(n_mem_train), n_mem_train)
    np.save(out_dir+"big_pred/"+ str(seed) +"_big_pred.npy", pred_big_im)
    np.save(out_dir+"pixels_pred/"+str(seed) +"_pixels_pred.npy", pred_vector)
    np.save(out_dir+"big_prob/"+ str(seed) +"_big_prob.npy", prob_big_im)
    cm = confusion_matrix(big_target_vector, pred_big_im)
    np.save(out_dir + "cm/" +  str(seed) + "_cm.npy", cm)
    print(cm)
    print("f1", f1_score(big_target_vector, pred_big_im))
    print("rec", recall_score(big_target_vector, pred_big_im))
    print("prec", precision_score(big_target_vector, pred_big_im))
    print("roc", roc_auc_score(big_target_vector, pred_big_im))
    if cluster ==0:
        from matplotlib import pyplot as plt
        plt.imsave(out_dir + "big_pred_seed_"+ str(seed) + ".png", np.reshape(pred_big_im, (im_size,im_size)), cmap="gray")
    else:
        print("Predictions are NOT saved as images, running on a cluster")

else:
    print("Not enough membrane pixels for training:", c_n_mem_pixels)
