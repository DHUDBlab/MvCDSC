"""
 Utility file for visualizing the data / loss curves
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import manifold


def visualize_data_tsne(Z, labels, num_clusters, title):
    '''
    TSNE visualization of the points in latent space Z
    :param Z: Numpy array containing points in latent space in which clustering was performed
    :param labels: True labels - used for coloring points
    :param num_clusters: Total number of clusters
    :param title: filename where the plot should be saved
    :return: None - (side effect) saves clustering visualization plot in specified location
    '''
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=fig.dpi)


