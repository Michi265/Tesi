#%matplotlib inline
import sys
import torch

import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)



def PCA_seaborn(labels,fc1_pca,fc2_pca):

    sns.set(style="darkgrid")
    #cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

    labels =labels.squeeze()
    #print (labels)

    sns.scatterplot(x=fc1_pca[:,0],y=fc1_pca[:,1],hue=labels,legend="full",palette=sns.color_palette())
    plt.figure()
    sns.scatterplot(fc2_pca[:,0], fc2_pca[:,1], labels,legend="full",palette=sns.color_palette())
    np.unique(labels)
    plt.show()

