
import torch

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)


def TSNE_seaborn(labels,fc1_embedded,fc2_embedded):


    sns.set()
    labels= labels.squeeze()

    sns.scatterplot(fc1_embedded[:, 0], fc1_embedded[:, 1], labels.squeeze(), legend="full",
                    palette=sns.color_palette())

    plt.figure()
    sns.scatterplot(fc2_embedded[:, 0], fc2_embedded[:, 1], labels.squeeze(), legend="full",
                    palette=sns.color_palette())
    np.unique(labels.squeeze())
    plt.show()
