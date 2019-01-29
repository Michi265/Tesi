import torch
import seaborn as sns
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select
import bokeh.palettes as palettes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Visualizer:
    def __init__(self, x, y, labels, palette='Category'):
        # Save a copy of everything.
        self.x = x
        self.y = y
        self.labels = labels

        # The list of available palettes.
        self.palettes = {
            'Viridis': palettes.Viridis10,
            'Magma': palettes.Magma10,
            'Category': palettes.Category10[10],
            'Plasma': palettes.Plasma10
        }

        # Setup the default palette and the mapping from labels to colors.
        self.default_palette = self.palettes[palette]
        self.colors = [self.default_palette[c] for c in labels]

        # IMPORTANT: this is how we get data into the plot, it will let us *change* the palette later.
        self.source = ColumnDataSource(data={'colors': self.colors, 'x': self.x, 'y': self.y, 'desc': labels})

    # This method can be passed to the Bokeh show() method to visualize the TSNE scatterplot and
    # the controls. For now, just the selector widget that allows user to select palette.
    def visualize(self, doc):
        # Make the figure and the scatterplot.
        tooltips = [('Label', '@desc')]
        p = figure(title="Visualization", toolbar_location='above', tooltips=tooltips)

        p.circle(x='x', y='y', size=5, line_color=None, fill_color='colors', alpha=1.0,
                 source=self.source, legend='desc')

        # Palette selection widget, plus callback.
        palette_select = Select(title="Color Palette:", options=list(self.palettes.keys()))
        palette_select.on_change('value', self.change_palette)

        # Add the selection widget plus scatterplot.
        doc.add_root(palette_select)
        doc.add_root(p)
        p.legend.click_policy="hide"
        p.legend.location = "top_right"

    # This method changes the palette.
    def change_palette(self, attr, old, new):
        # Compute the new mapping from lab
        new_palette = self.palettes[new]
        self.colors = [new_palette[l] for l in self.labels]

        # The patch() method of the DataColumnSource lets us update data in plots.
        self.source.patch({'colors': [(slice(len(self.colors)), self.colors)]})

def PCA_bokeh(labels,fc1,fc2):

    fc1 = fc1.detach().numpy()
    fc2 = fc2.detach().numpy()
    labels = labels.detach().numpy()
    labels = labels.reshape(-1, 1)
    pca1 = PCA(n_components=2)
    pca1.fit(fc1)

    pca2 = PCA(n_components=2)
    pca2.fit(fc2)

    fc1_pca = pca1.transform(fc1)
    fc2_pca = pca2.transform(fc2)

    output_notebook()
    x = fc2_pca[:,0]
    y = fc2_pca[:,1]
    vis = Visualizer(x, y, labels.squeeze())
    show(vis.visualize)


def TSNE_seaborn(labels,fc1,fc2):
    fc1 = fc1.detach().numpy()
    fc2 = fc2.detach().numpy()
    labels = labels.detach().numpy()
    labels = labels.reshape(-1, 1)

    fc1_embedded = TSNE(n_components=2).fit_transform(fc1)
    fc2_embedded = TSNE(n_components=2).fit_transform(fc2)

    sns.set()
    labels= labels.squeeze()

    sns.scatterplot(fc1_embedded[:, 0], fc1_embedded[:, 1], labels.squeeze(), legend="full",
                    palette=sns.color_palette())

    plt.figure()
    sns.scatterplot(fc2_embedded[:, 0], fc2_embedded[:, 1], labels.squeeze(), legend="full",
                    palette=sns.color_palette())
    np.unique(labels.squeeze())
    plt.show()


def TSNE_bokeh(labels,fc2):
    fc2 = fc2.detach().numpy()
    labels = labels.detach().numpy()
    labels = labels.reshape(-1, 1)

    fc2_embedded = TSNE(n_components=2).fit_transform(fc2)

    output_notebook()
    x = fc2_embedded[:,0]
    y = fc2_embedded[:,1]
    vis = Visualizer( x, y, labels.squeeze())
    show(vis.visualize)
