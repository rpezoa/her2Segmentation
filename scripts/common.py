import numpy as np
import random as rd
from matplotlib import pyplot as plt
import matplotlib.patches as patches

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


def indexFromMat2Lineal(the_list,n_cols):
    # Obtiene indices lineales a partir de indices de
    # una matriz
    indices = [None] * len(the_list)
    for i,elem in enumerate(the_list):
        indices[i] = elem[0]*n_cols + elem[1]
    return indices

def training_data(X,y,nX,ny):
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

def patch_square(position, width, height,color):
    #print("pos:", position)
    return patches.Rectangle(
        position,
        width,
        height,
        fill=False,
        edgecolor=color,
	lw=4)

def write_sub_im(im, coordinates, out_dir, n_patches, seed, patch_width, patch_height, color, name, mem_number=None, cMap=None,my_dpi=96):
    from skimage import img_as_ubyte
    shift_x = (patch_width - 1)/2
    shift_y =  (patch_height - 1) /2
    fig = plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi = my_dpi)
    plt.axis("off")
    ax = fig.add_subplot(111)

    #Mostramos la imagen y usamos add_patch 
    #con la función para agregar el cuadrado 
    ax.imshow(im,cmap="gray")
    
    if len(im.shape) == 2:
        im=img_as_ubyte(im)
        print("yes")
        print(im[:10,:10])
        
    for c, center in enumerate(coordinates):
        patch_position = center
        ax.add_patch(patch_square((patch_position[1] - shift_y, patch_position[0] - shift_x), patch_width, patch_height,color))
        #ax.annotate( "(" + str(patch_position[0]) + "," + str(patch_position[1]) + ")",xy=(patch_position[1], patch_position[0]), fontsize=10, color="g")
        #circ = patches.Circle((patch_position[1], patch_position[0]),3, color="m")
        #ax.add_patch(circ)
        if mem_number is not None:
            ax.annotate( str(mem_number[c]),xy=(patch_position[1], patch_position[0] - shift_y), fontsize=12, color="m")
    title = out_dir + name + "_" + str(seed) + "_seed_"+ str(n_patches) + "_patches.png"
    if cMap:
        fig.savefig(title, dpi=my_dpi, bbox_inches='tight')
    else:
        fig.savefig(title, dpi=my_dpi, bbox_inches='tight', cmap="gray")
                
