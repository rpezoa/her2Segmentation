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
parser.add_argument('-suf', '--suffix', help="Feature Folder suffix")
parser.add_argument('-score','--score', help="SVM Performance Score")
parser.add_argument('-is','--image_size', help="Image size (square image)", type=int)
parser.add_argument('-btp','--big_target_path', help="Big Target Path")
parser.add_argument('-tp','--target_path', help="Target Path")
parser.add_argument('-feat_path', '--features_path', help="Feature Path")
parser.add_argument('-bp', '--big_path', help="Big scaled feature Path")
parser.add_argument('-seed','--seed', help="Seed", type=int)
args = parser.parse_args()

args.input_path = args.input_path + '/' if args.input_path[-1] != '/' else args.input_path
args.output_path = args.output_path + '/' if args.output_path[-1] != '/' else args.output_path

seed = int(args.seed)
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
print ("Directory of Data: %s" % args.input_path )
print ("Output Directory: %s" % args.output_path)
print ("Image to Process: %s" % args.image )

baseDir = args.input_path
imgDir =  baseDir + 'img/'
maskDir = baseDir + 'masks/'
featDir = baseDir + 'features/'
baseOutDir = baseDir + 'results/'
pyDir = args.output_path

image_name = args.image.split('.')[0]
print ("Feature Set Directory: %s" % args.input_path )
im_name =  args.image

if args.suffix:
    suffix = args.suffix
else:
    suffix = ''

outPyDir = pyDir + im_name.split(".")[0] +"/" + suffix + "/halton_patches/"

if not os.path.exists(outPyDir):
    os.makedirs(outPyDir)
print("Saving files in: ", outPyDir)

if not os.path.exists(outPyDir + "/big_pred"):
    os.makedirs(outPyDir + "big_pred")
if not os.path.exists(outPyDir + "/pixels_pred"):
    os.makedirs(outPyDir + "pixels_pred")
if not os.path.exists(outPyDir + "/times"):
    os.makedirs(outPyDir + "times")
if not os.path.exists(outPyDir + "/cm"):
    os.makedirs(outPyDir + "cm")
if not os.path.exists(outPyDir + "/big_prob"):
    os.makedirs(outPyDir + "big_prob")

print("\n" + "%"*20 +" Seed: " +str(seed) + "%"*20 )
big_scaled_feat_matrix = np.load(big_path)
big_target_vector = np.load(big_target_path)


# separating classes
mask_c1 = big_target_vector == 1
mask_c2 = big_target_vector == 0

labels_c1 = big_target_vector[mask_c1]
labels_c2 = big_target_vector[mask_c2]
features_c1 = big_scaled_feat_matrix[mask_c1]
features_c2 = big_scaled_feat_matrix[mask_c2]

n1 = mask_c1.sum()
n2 = mask_c2.sum()

rand_index_c1 = rd.sample(range(n1), 250)
rand_index_c2 = rd.sample(range(n2), 650)

current_X1 =features_c1[rand_index_c1,:]
current_X2 =features_c2[rand_index_c2,:]

current_y1  = labels_c1[rand_index_c1]
current_y2  = labels_c2[rand_index_c2]


current_X = np.concatenate((current_X1, current_X2), axis=0)
current_y = np.concatenate((current_y1, current_y2), axis=0)

nX,ny= current_y.sum(), current_y.sum()
print(ny)


n_mem_train = current_y.sum()
print("current_X.shape",current_X.shape)
print("curret_y.shape", current_y.shape)
np.save(outPyDir + "n_mem_train_seed_" + str(seed) + "_" + str(n_mem_train), n_mem_train)
c_n_mem_pixels = (current_y==1).sum()

print('Current mem pixels: ', c_n_mem_pixels)
print('Current non-mem pixels:', (current_y==0).sum())
print("N features:", current_X.shape[1])
print("current_X.mean()", current_X.mean())

N = current_X.shape[0]
if c_n_mem_pixels > 3:
	tuned_parameters = [{'kernel': [kernel], 'gamma': gamma_range,'C': C_range}]
	clf = GridSearchCV(SVC(class_weight="balanced", probability=True), tuned_parameters, cv=k, scoring=score,n_jobs = -1, verbose=1)
	print("******** Total current X:", current_X.shape, "***********")
	x_train, x_test, y_train, y_test = train_test_split(current_X, current_y, test_size=0.2, random_state=0)
	try:
		print("::::::: Training with ", x_train.shape, ":::::::::::::")
		t1=time.time()
		clf.fit(x_train, y_train)
		t2=time.time()
		np.save(outPyDir + "times/training_" + str(seed) + ".npy", t2-t1) 
		print("SVM fitting time:", t2-t1)
		if not os.path.exists(outPyDir + "pkl/" + str(seed)):
			os.makedirs(outPyDir + "pkl/" + str(seed))
		if clf is not None:
			joblib.dump(clf, outPyDir + 'pkl/' + str(seed) + '/clf_' +str(seed) + '.pkl')
		#------------------------------------------------------  
		# Generating predictions and saving the predictions
		best_estimator = clf.best_estimator_
		pred_vector = best_estimator.predict(current_X)
		pred_big_im = best_estimator.predict(big_scaled_feat_matrix)
		prob_big_im = best_estimator.predict_proba(big_scaled_feat_matrix)
		np.save(outPyDir+"big_pred/"+ str(seed) +"_big_pred.npy", pred_big_im)
		np.save(outPyDir+"pixels_pred/"+str(seed) +"_pixels_pred.npy", pred_vector)
		np.save(outPyDir+"big_prob/"+ str(seed) +"_big_prob.npy", prob_big_im)
		cm = confusion_matrix(big_target_vector, pred_big_im)
		np.save(outPyDir + "cm/" +  str(seed) + "_cm.npy", cm)
		print(cm)
		print("f1", f1_score(big_target_vector, pred_big_im))
		print("rec", recall_score(big_target_vector, pred_big_im))
		print("prec", precision_score(big_target_vector, pred_big_im))
		print("roc", roc_auc_score(big_target_vector, pred_big_im))
		plt.imsave(outPyDir + "big_pred_seed_"+ str(seed) + ".png", np.reshape(pred_big_im, (im_size,im_size)), cmap="gray")
	except:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		print("Error with SVM classifier generation: ", exc_value)
else:
    print("Not enough membrane pixels for SVM training:", c_n_mem_pixels)
