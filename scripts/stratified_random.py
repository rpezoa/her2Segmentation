import sys
import os
sys.path.append('/home/rpezoa/svm/')
#import warnings
#warnings.filterwarnings('ignore')
import numpy as np
import time
from common import write_sub_im, indexFromMat2Lineal, get_neighbors_index
from matplotlib import pyplot as plt
import argparse
import random as rd

parser = argparse.ArgumentParser(description='Testing images.')
parser.add_argument('input_path', help='Directory of features')
parser.add_argument('output_path', help='Directory to save results' )
parser.add_argument('image',help='Image name without extension')
parser.add_argument('-seed','--seed', help="Seed", type=int)
parser.add_argument('-feat_path', '--features_path', help="Feature Path")
parser.add_argument('-btp','--big_target_path', help="Big Target Path")

args = parser.parse_args()

args.input_path = args.input_path + '/' if args.input_path[-1] != '/' else args.input_path
args.output_path = args.output_path + '/' if args.output_path[-1] != '/' else args.output_path

seed = int(args.seed)
if args.features_path:
    feat_path = args.features_path
if args.big_target_path:
    big_target_path = args.big_target_path

print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Seed:", seed, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Directory of Data: %s" % args.input_path )
print ("Output Directory: %s" % args.output_path)
print ("Image to Process: %s" % args.image )

baseDir = args.input_path
imgDir =  baseDir + 'img/'
maskDir = baseDir + 'masks/'
featDir = baseDir + 'features/'
baseOutDir = baseDir + 'results/'
output_dir = args.output_path

image_name = args.image.split('.')[0]
print ("Feature Set Directory: %s" % args.input_path )
im_name =  args.image


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving files in: ", output_dir)


if not os.path.exists(output_dir + "/feat_vectors"):
    os.makedirs(output_dir + "feat_vectors")
if not os.path.exists(output_dir + "/target_vectors"):
    os.makedirs(output_dir + "target_vectors")
if not os.path.exists(output_dir + "/idx_training_pixels"):
    os.makedirs(output_dir + "idx_training_pixels")

training_size = 13005
big_target_vector = np.load(big_target_path) 
big_scaled_feat_matrix = np.load(feat_path)

## Getting stratified random indices

mask_0 = big_target_vector == 0
mask_1 = big_target_vector == 1
idx_0 = np.where(mask_0 == True)[0] #indices of pixels with target equal 0 (non-mem) 
idx_1 = np.where(mask_1 == True)[0] #indices of pixels with target equal 1 (mem)
print("idx_0.shape", idx_0.shape, "idx_1.shape", idx_1.shape)

n_0 = mask_0.sum()
n_1 = mask_1.sum()
print("non-mem:", n_0, "mem:", n_1)
y_0 = big_target_vector[mask_0]
y_1 = big_target_vector[mask_1]

X_0 = big_scaled_feat_matrix[mask_0] #only non-mem
X_1 = big_scaled_feat_matrix[mask_1] #only mem
print("X_0.shape, X_1.shape", X_0.shape, X_1.shape)

rd.seed(seed)
rd_idx_0 = rd.sample(range(n_0),int(0.95 * training_size))
rd_idx_1 = rd.sample(range(n_1),int(0.05 * training_size))


X_0 = X_0[rd_idx_0,:]
X_1 = X_1[rd_idx_1,:]

y_0 = y_0[rd_idx_0]
y_1 = y_1[rd_idx_1]

new_idx_0 = idx_0[rd_idx_0]
new_idx_1 = idx_1[rd_idx_1]

rd_X = np.concatenate((X_0,X_1),axis=0)
rd_y = np.concatenate((y_0,y_1),axis=0)
print("X.shape:", rd_X.shape, "y.shape", rd_y.shape)

        
np.save(output_dir+"feat_vectors/"+str(seed)+".npy", rd_X)
np.save(output_dir+"target_vectors/"+str(seed)+".npy",rd_y)
np.save(output_dir+"idx_training_pixels/"+str(seed)+ "_0.npy", new_idx_0)
np.save(output_dir+"idx_training_pixels/"+str(seed)+ "_1.npy", new_idx_1)



