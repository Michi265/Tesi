# mylibrary.py
import seaborn as sns
#%matplotlib inline
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable



import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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


if __name__ == '__main__':
    '''il file è stato eseguito direttamente'''