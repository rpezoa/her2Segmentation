import numpy as np
import random as rd
#####################################################
#           Get Neightbors Index
#####################################################

def get_neighbors_index(centers,shape,d):
    if type(centers) == int:
        centers = [centers]


    index = np.arange(shape[0]*shape[1]).reshape(shape)
    index_flatten = index.flatten()

    flatten_list = []

    for center in centers:

        if center < shape[0]*shape[1] - 1:
            unflatten_index = np.unravel_index(center,shape)

            i,j = unflatten_index

            i_slice_left = i-d if i-d >= 0 else 0
            i_slice_right = i+d+1 if i+d+1 <= index.shape[0] -1  else index.shape[0] - 1
            j_slice_left = j-d if j-d >= 0 else 0
            j_slice_right = j+d+1 if j+d+1 <= index.shape[1] -1  else index.shape[1] - 1

            flatten_index = index[i_slice_left:i_slice_right, j_slice_left:j_slice_right].flatten()

            flatten_list.append(flatten_index)

    filtered = np.hstack(flatten_list)
    return filtered

def training_data(X,y,nX,ny):
	mask_1 = y == 1
	mask_2 = y == 0

	labels_1 = y[mask_1] 
	labels_2 = y[mask_2]
	feat_1 =X[mask_1]
	feat_2 = X[mask_2]

	n_1 = mask_1.sum()
	n_2 = mask_2.sum()

	ind_1 = rd.sample(range(n_1),nX)
	ind_2 = rd.sample(range(n_2), ny)


	X_1 =feat_1[ind_1,:]
	X_2 =feat_2[ind_2,:]

	y_1  = labels_1[ind_1]
	y_2  = labels_2[ind_2]
	new_X = np.concatenate((X_1, X_2), axis=0)
	new_y = np.concatenate((y_1, y_2), axis=0)
	return new_X, new_y
