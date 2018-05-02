import sys
import os
sys.path.append('/home/rpezoa/svm/')
#import warnings
#warnings.filterwarnings('ignore')
import numpy as np
import ghalton
import time
from common import write_sub_im, indexFromMat2Lineal, get_neighbors_index
from matplotlib import pyplot as plt
import argparse
import random as rd

parser = argparse.ArgumentParser(description='Testing images.')
parser.add_argument('input_path', help='Directory of features')
parser.add_argument('output_path', help='Directory to save results' )
parser.add_argument('image',help='Image name without extension')
parser.add_argument('-s', '--size', help="Patch Size", type=int)
parser.add_argument('-is','--image_size', help="Image size (square image)", type=int)
parser.add_argument('-kp','--kpatches', help="Number of selected patches", type=int)
parser.add_argument('-np','--npatches', help="Number of patches", type=int)
parser.add_argument('-seed','--seed', help="Seed", type=int)
parser.add_argument('-feat_path', '--features_path', help="Feature Path")
parser.add_argument('-th', '--threshold', help="Threshold for color channel", type=float)
parser.add_argument('-feat_names', '--features_names', help="Features Name Path")
parser.add_argument('-channel_path', '--channel_path', help="Channel path for GS estimation")
parser.add_argument('-channel_name', '--channel_name', help="Channel name")
parser.add_argument('-t_max', '--t_max', help="Max threshold for patch filtering")
parser.add_argument('-btp','--big_target_path', help="Big Target Path")

args = parser.parse_args()

args.input_path = args.input_path + '/' if args.input_path[-1] != '/' else args.input_path
args.output_path = args.output_path + '/' if args.output_path[-1] != '/' else args.output_path
size = args.size if args.size else size

if args.threshold:
    threshold = args.threshold
if args.npatches:
    n_patches = args.npatches
if args.kpatches:
    k_patches = args.kpatches
if args.image_size:
    image_size = args.image_size
seed = int(args.seed)
if args.t_max:
    t_max = float(args.t_max)
n_patch = int(args.npatches)
if args.features_path:
    feat_path = args.features_path
if args.channel_path:
    channel_path = args.channel_path
if args.channel_name:
    channel_name = args.channel_name
if args.features_names:
    feat_names = args.features_names
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
if not os.path.exists(output_dir + "/big_pred"):
    os.makedirs(output_dir + "big_pred")
if not os.path.exists(output_dir + "/idx_training_pixels"):
    os.makedirs(output_dir + "idx_training_pixels")


big_target_vector = np.load(big_target_path) 
big_im = np.zeros((1000,1000,3)) #cambiar esto!!

#big_scaled_feat_matrix, feat_names, big_feat_matrix = exp_f.get_features(bigDir)
big_scaled_feat_matrix = np.load(feat_path)
b_rows, b_cols, b_dim = big_im.shape
n_pixels = b_rows * b_cols

# GET CHANNEL FOR GS GENERATION
channel = np.load(channel_path)
channel_im = np.reshape(channel,(image_size,image_size))
plt.imsave(output_dir + channel_name + "_channel.png", channel_im, cmap="gray")

# Revisar si es channel o 1-channel
channel_mask = channel <= threshold
mask_num = channel[channel_mask]
print("mask_num", mask_num)
mask_im = np.reshape(channel_mask, (image_size, image_size))
plt.imsave(output_dir + channel_name + "_channel_mask_thres_" + str(threshold) + ".png", mask_im, cmap="gray")

##############################
# GENERATE 2D HALTON PATCHES ([0,1] RANGE)
# ACCORDING TO SEED
sequencer = ghalton.GeneralizedHalton(2,0)
for i in range(seed+1):
    print("sequencer i: ", i)
    points = sequencer.get(n_patches)
    
n_iter=seed
titles = str(n_iter)+"_iter_"+str(n_patches)+"_patches"

#############################
# HALTON POINTS IN RANGE OF IMAGE SIZE 

#Se generan tantos Halton points como numero de patches
#Recorro todos los halton points para sacar su primera y segunda
# coordenada y los almaceno en una lista
l1=[]
l2=[]
the_list = [None] * n_patches
for i in range(n_patches):
    a= int(points[i][0]*image_size)
    b= int(points[i][1]*image_size)
    l1.append(a)
    l2.append(b)
    the_list[i] = [a,b]
###########################
# READING IMAGE AND SAVING
# IMAGES WITH PATCHES
image=plt.imread(imgDir + args.image)
patch_width = size*2 + 1
patch_height = size*2 + 1

write_sub_im(image, the_list, output_dir, n_patches, seed, patch_width, patch_height, "red", "im")
write_sub_im(mask_im, the_list, output_dir, n_patches, seed, patch_width, patch_height, "red", channel_name + "_mask")

##########################
# HALTON POINTS WITH LINEAL INDICES
the_center = indexFromMat2Lineal(the_list, image_size)       
print("the_center", the_center)
mem_estim = np.zeros(len(the_center))
mem_gs = np.zeros(len(the_center))
f1_mat = np.zeros(len(the_center))

###########################
# COUNTING ESTIMATED PIXELS AND
# GOLD STANDAR PIXELS PER PATCH
for idx, j in enumerate(the_center):
    # Getting neighbors of the lineal index
    the_neigh =  get_neighbors_index(j, big_im.shape[0:2], size)
    n_neigh = len(the_neigh)
    count = 0 # contador para estimado
    count_mask = 0 # contador para gold standard
    # Getting the label of each pixel of "the_neigh"
    # and the amount of "mem pixels" (according y channel)
    for i,tn in enumerate(the_neigh):
        if channel_mask[tn]:
            count = count + 1
    # If the amount of "mem pixels" is more than 3.5% of the neighborhood,
    # then the patch is selected
        if big_target_vector[tn]:
            count_mask = count_mask + 1
    mem_estim[idx] = count
    mem_gs[idx] = count_mask

#########################
# FILTERING ACCORDING Y CHANNEL THRESHOLD
new_center = []
new_list = []
y_number = []
gs_number = []
idx_sel = []
for idx, j in enumerate(the_center):
    # Getting neighbors of the lineal index
    the_neigh =  get_neighbors_index(j, big_im.shape[0:2], size)
    n_neigh = len(the_neigh)
    count = 0
    count_mask = 0
    # Getting the label of each pixel of "the_neigh"
    # and the amount of "mem pixels" (according y channel)
    if mem_estim[idx] <= t_max * n_neigh: # Aca se filtran patches, dejando solo los que tiene menos de 15% de mem
        new_center.append(j) # adding the lineal index of the patch
        new_list.append(the_list[idx]) # adding the 2D index of the patch
        gs_number.append(count_mask)
        idx_sel.append(idx)
        y_number.append(mem_estim[idx])
    else:
        y_number.append(-1)

new_mem_estim = []
new_mem_gs = []
for i, index in enumerate(idx_sel):
    new_mem_estim.append(mem_estim[index])
    new_mem_gs.append(mem_gs[index])

y_number = np.asarray(y_number)
new_n_patches = len(new_center)
image=plt.imread(imgDir + args.image)
patch_width = size*2 + 1
patch_height = size*2 + 1
#com.write_sub_im(image, new_list, output_dir, new_n_patches, seed, patch_width, patch_height, "red", "filtered")
#com.write_sub_im(mask_im, new_list, output_dir, new_n_patches, seed, patch_width, patch_height, "red", "filtered_" +channel_name+ "_mask", mem_number=new_mem_estim, cMap="gray")
#com.write_sub_im(mask, new_list, output_dir, new_n_patches, seed, patch_width, patch_height, "red", "filtered_mask", mem_number=new_mem_gs,cMap="gray")


####################################################
# Recorro cada halton point y genero el patch (cuyos indices estan almacenados en random_index) 
print("Seed:", seed, "new_n_patches", new_n_patches)
X_list = [None] * new_n_patches
y_list = [None] * new_n_patches
indices_selected = [None] * k_patches
sel_centers =[]
sel_patches_list = []
print(" :::::::: Seleting patch 1")

##############################
# SELECTING K PATCHES WITH THE MAX "AMOUNT OF MEM PIX"
y_max_idx = y_number.argsort()
#print("y_max_idx", y_max_idx)
#print("new_center", new_center)

es_total = 0
es_number = []
gs_total = 0
gs_number = []
pixels_idx = np.array([])

if new_n_patches >= k_patches:
	for i, idx in enumerate(y_max_idx[-k_patches:]):
		random_index = get_neighbors_index(the_center[idx], big_im.shape[0:2], size)
		pixels_idx = np.append(pixels_idx,random_index)
		X_list[i] = big_scaled_feat_matrix[random_index,:]
		y_list[i] = big_target_vector[random_index]
		gs_n = y_list[i].sum()
		es_n = channel_mask[random_index].sum()
		gs_total = gs_total + gs_n
		es_total = es_total + es_n
		gs_number.append(gs_n)
		es_number.append(es_n)
		sel_centers.append(the_center[idx])
		sel_patches_list.append(the_list[idx])
	np.save(output_dir+str(seed)+ "_gs_" +str(gs_total) + ".npy", gs_total)
	np.save(output_dir+"idx_training_pixels/"+str(seed)+"_idx_0.npy", pixels_idx)

	X_mat = X_list[0]
	y_mat = y_list[0]
	for i in range(k_patches -1):
		X_mat = np.append(X_mat,X_list[i+1], axis=0)
		y_mat = np.append(y_mat, y_list[i+1], axis=0)

	print("X_mat.shape", X_mat.shape)
	print("y_mat.sum()", y_mat.sum())
	np.save(output_dir+"feat_vectors/"+str(seed)+ ".npy", X_mat)
	np.save(output_dir+"target_vectors/"+str(seed)+ ".npy", y_mat)
	print("y_mat.shape", y_mat.shape)
#	com.write_sub_im(image, sel_patches_list, output_dir, len(sel_patches_list),seed, patch_width, patch_height, "green", "selected")
#	com.write_sub_im(mask, sel_patches_list, output_dir, len(sel_patches_list),seed, patch_width, patch_height, "green", "mask_selected", mem_number=gs_number)
#	com.write_sub_im(mask_im, sel_patches_list, output_dir, len(sel_patches_list),seed, patch_width, patch_height, "green", channel_name + "_mask_selected", mem_number=es_number)

	n_patches_pixels = y_mat.shape[0]
	n_class_1 = y_mat.sum()
	n_class_2 = n_patches_pixels - n_class_1
	print("n_class_1", n_class_1, "n_class_2", n_class_2)
	n_min_class_1 = 0.05 * X_mat.shape[0] # around 650, because X_mat.shape 
						  # can be less than 13005 
		                                  # considering some overlapping
	new_size = size * 2 
	patch_width = new_size*2 + 1
	patch_height = new_size*2 + 1

	if n_class_1 < n_min_class_1:
		print("Extending patches ...")
		extended_patches = get_neighbors_index(sel_centers, big_im.shape[0:2], new_size)
		final_patches = big_scaled_feat_matrix[extended_patches,:]
		print("final_patches.shape", final_patches.shape)
		final_target = big_target_vector[extended_patches]

		mask_class_1 = final_target == 1
		print("mask_class_1.sum()", mask_class_1.sum())
		class_1_target = final_target[mask_class_1]
		class_1_features = final_patches[mask_class_1]
		class_1_idx = extended_patches[mask_class_1]
		print("Extending patches ... ", mask_class_1.sum(), "mem ...")
		print("Selecting only 5%")
		
		n_selec = n_min_class_1 - n_class_1
		print("Only taking", int(n_selec), "mem pixels") 
		rand_idx_class_1 = rd.sample(range(mask_class_1.sum()), int(n_selec))
		real_idx = class_1_idx[rand_idx_class_1]
		  
		final_patches = np.concatenate((X_mat, big_scaled_feat_matrix[real_idx]), axis=0)
		final_target = np.concatenate((y_mat, big_target_vector[real_idx]), axis=0)
		np.save(output_dir+"idx_training_pixels/"+str(seed) + "_idx_1.npy", real_idx)

		# Se agregaron solo los pixeles de la clase membrana
		np.save(output_dir+"feat_vectors/"+str(seed)+ ".npy", final_patches)
		np.save(output_dir+"target_vectors/"+str(seed)+ ".npy", final_target)
		print("New amount of class 1:", final_target.sum(), "final_patches.shape", final_patches.shape)
#		com.write_sub_im(mask_im, sel_patches_list, output_dir, len(sel_patches_list),seed, patch_width, patch_height, "green", "extended")

	elif n_class_1 > n_min_class_1:
		n_del_class_1 =  n_class_1 - (0.05 * X_mat.shape[0]) 
		print("Removing", int(n_del_class_1)  ," point of class 1")
		y_class_1_idx = np.where(y_mat == 1)[0]
		rand_y_idx_class_1 = rd.sample(range(n_class_1), int(n_del_class_1))
		real_idx = y_class_1_idx[rand_y_idx_class_1]

		new_X = np.delete(X_mat,real_idx, axis=0)
		new_y = np.delete(y_mat, real_idx, axis=0)
		print("new_X.shape", new_X.shape)
		print("new_y.sum()", new_y.sum())
		np.save(output_dir+"feat_vectors/"+str(seed)+ ".npy", new_X)
		np.save(output_dir+"target_vectors/"+str(seed)+ ".npy", new_y)
		print("new_y.shape", new_y.shape)
else:
	print(":::::::::: Not enough filtered patches,", new_n_patches, "filtered patches ::::::::::")

