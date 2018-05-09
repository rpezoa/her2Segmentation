import sys
import os
sys.path.append('/home/rpezoa/svm/')
#import warnings
#warnings.filterwarnings('ignore')
import numpy as np
import time
from common import training_data
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import argparse

import random as rd
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (recall_score,  precision_score, f1_score, roc_auc_score, 
                             make_scorer, confusion_matrix)
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
parser.add_argument('-feat_path', '--features_path', help="Feature Path")
parser.add_argument('-bp', '--big_path', help="Big scaled feature Path")
parser.add_argument('-seed','--seed', help="Seed", type=int)
parser.add_argument('-clf','--classifier', help="Seed")
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

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print("Saving files in: ", out_dir)

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

print("\n" + "%"*20 +" Seed: " +str(seed) + "%"*20 )
big_scaled_feat_matrix = np.load(big_path)
big_target_vector = np.load(big_target_path)

current_X = np.load(feat_path)
current_y = np.load(target_path)

nX,ny= current_y.sum(), current_y.sum()
print(ny)

if under_sampling == 1:
# Here, it is the undersampling
    current_X, current_y = training_data(current_X, current_y,ny,ny )
    print("Features vectors undersampled")
n_mem_train = current_y.sum()
print("current_X.shape",current_X.shape)
print("curret_y.shape", current_y.shape)
np.save(out_dir + "n_mem_train_seed_" + str(seed) + "_" + str(n_mem_train), n_mem_train)
c_n_mem_pixels = (current_y==1).sum()

print('Current mem pixels: ', c_n_mem_pixels)
print('Current non-mem pixels:', (current_y==0).sum())
print("N features:", current_X.shape[1])
print("current_X.mean()", current_X.mean())

N = current_X.shape[0]

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
    
def deep_l(X,y):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import metrics

    print("Deep Learning :::::: X,y", X.shape, y.shape) 
    clf = Sequential()
    clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 40))
    clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # https://datascience.stackexchange.com/questions/13746/how-to-define-a-custom-performance-metric-in-keras
    clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    clf.fit(x_train, y_train, batch_size = 10, nb_epoch = 1000)
    pred_big_im = clf.predict(big_scaled_feat_matrix)
    pred_big_im = (pred_big_im > 0.5)
    return pred_big_im, pred_big_im, pred_big_im

def random_f(X,y,big_X):
    from sklearn.ensemble import RandomForestClassifier
    print("Random forest :::::: X,y", X.shape, y.shape) 
    clf = RandomForestClassifier(n_estimators=25, criterion="entropy")
    t1=time.time()
    clf.fit(x_train, y_train)
    t2=time.time()
    pred_vector = clf.predict(X)
    pred_big_im = clf.predict(big_X)
    return pred_vector, pred_big_im, pred_big_im

def kNN(X,y,big_X):
    from sklearn.neighbors import KNeighborsClassifier
    print("KNN :::::: X,y", X.shape, y.shape) 
    clf = KNeighborsClassifier()
    print("::::::: Training with ", x_train.shape, ":::::::::::::")
    t1=time.time()
    clf.fit(x_train, y_train)
    t2=time.time()
    print("SVM fitting time:", t2-t1)
    pred_vector = clf.predict(X)
    pred_big_im = clf.predict(big_X)
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

def extra_trees(X,y,big_X):
    from sklearn.ensemble import ExtraTreesClassifier
    print("Extra Trees :::::: X,y", X.shape, y.shape) 
    clf = ExtraTreesClassifier(n_estimators=25)
    print("::::::: Training with ", x_train.shape, ":::::::::::::")
    t1=time.time()
    clf.fit(x_train, y_train)
    t2=time.time()
    print("SVM fitting time:", t2-t1)
    pred_vector = clf.predict(X)
    pred_big_im = clf.predict(big_X)
    return pred_vector, pred_big_im, pred_big_im


if c_n_mem_pixels > 3:
    x_train, x_test, y_train, y_test = train_test_split(current_X, current_y, test_size=0.2, random_state=0)
    if c == "svm":
        pred_vector, pred_big_im, prob_big_im = svm_c(x_train,y_train, big_scaled_feat_matrix)
    elif c == "deep":
        pred_vector, pred_big_im, prob_big_im  = deep_l(x_train, y_train)
    elif c == "rf":
        pred_vector, pred_big_im, prob_big_im = random_f(x_train,y_train, big_scaled_feat_matrix)
    elif c == "knn":
        pred_vector, pred_big_im, prob_big_im = kNN(x_train,y_train, big_scaled_feat_matrix)
    elif c == "xgboost":
        pred_vector, pred_big_im, prob_big_im = xgboost(x_train, y_train, big_scaled_feat_matrix)
    elif c == "extra_trees":
        pred_vector, pred_big_im, prob_big_im = extra_trees(x_train,y_train, big_scaled_feat_matrix)



    
    # Generating predictions and saving the predictions
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
    plt.imsave(out_dir + "big_pred_seed_"+ str(seed) + ".png", np.reshape(pred_big_im, (im_size,im_size)), cmap="gray")

else:
    print("Not enough membrane pixels for SVM training:", c_n_mem_pixels)
