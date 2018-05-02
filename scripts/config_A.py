import numpy as np
#######################################################################################
# SVM parameters
svm_type = "C"
kernel = 'rbf'
k = 5
C_r = [-3, 3]
C_step = 0.5
g_r = [-3, 3]
g_step = 0.5
C_range = 10. ** np.arange(C_r[0], C_r[1],C_step)
gamma_range = 10. ** np.arange(g_r[0], g_r[1],g_step)
# score can be: roc_auc, accuracy, recall, precision, f1, average_precision
#score = 'average_precision'
score="f1"
test_size = 0.2
rs = 0
