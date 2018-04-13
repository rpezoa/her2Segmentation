import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# getting data
big_path = "/home/rpezoa/experiment_data/big_2+/features/"
big_target_path = "/home/rpezoa/experiment_data/big_2+/labels/"
X  = np.load(big_path + "2+_1_hsv.npy")
y = np.load(big_target_path + "2+_1.npy")

# separating classes
mask_c1 = y == 1
mask_c2 = y == 0

labels_c1 = y[mask_c1] 
labels_c2 = y[mask_c2]
features_c1 = X[mask_c1]
features_c2 = X[mask_c2]

n1 = mask_c1.sum()
n2 = mask_c2.sum()

rand_index_c1 = rd.sample(range(n1), 80)
rand_index_c2 = rd.sample(range(n2), 20)

current_X1 =features_c1[rand_index_c1,:]
current_X2 =features_c2[rand_index_c2,:]

current_y1  = labels_c1[rand_index_c1]
current_y2  = labels_c2[rand_index_c2]


current_X = np.concatenate((current_X1, current_X2), axis=0)
current_y = np.concatenate((current_y1, current_y2), axis=0)


X_2d = current_X[:,0:2]
y_2d = current_y[:]

print("X_2s and y_2s shapes", X_2d.shape, y_2d.shape) 
# scaling
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#X_2d = scaler.fit_transform(X_2d)

# SVM
C_range = np.logspace(1, 10, 13)
gamma_range = np.logspace(1, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs = -1, verbose=10)
grid.fit(X, y)
print("training set size:", X.shape)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))


# visualization
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()
