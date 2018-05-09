import numpy as np
import random as rd

def under_sampling(X,y,nX,ny):
	# It returns undersampled data
	mask_1 = y == 1 # class 1 is smallest than class 2
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
