# CS4412 : Data Mining
# Kennesaw State University

"""Basic structure for the MNIST handwritten digits dataset, for use
in this project.

"""

import numpy                          # a library for matrix math
from matplotlib import pyplot         # for plotting
from mpl_toolkits.axes_grid1 import ImageGrid # for drawing images
import pickle                         # reading/writing python objects
import random                         # random number generator
from collections import defaultdict   # dictionary with defaults

class Mnist:
    def __init__(self,mnist_dir="mnist",seed=0):
        self._seed = seed   # random number seed
        self._dim = (28,28) # dimension of images (28x28)
        # read in the dataset
        with open("%s/mnist-train-labels" % mnist_dir,"rb") as f: 
            self._labels = pickle.load(f)
        with open("%s/mnist-train-images" % mnist_dir,"rb") as f: 
            self._images = pickle.load(f)
        self._N = len(self._labels) # size of dataset

    def images(self):
        return self._images

    """Shrinks the dataset by a factor (default .5).  this is used to
    speed things up for testing. 

    """
    def shrink(self,factor=0.5):
        self._N = int(factor*self._N)
        self._labels = self._labels[:self._N]
        self._images = self._images[:self._N]

    """A must-link constraint is a pair (i,j) where i and j are 
    dataset indices and where the known label of i and j are the same.

    """
    def _simulate_must_link(self):
        while True:
            i,j = random.sample(range(self._N),2)
            if i > j: i,j = j,i
            if self._labels[i] == self._labels[j]: break
        return (i,j)

    """A must-not-link constraint is a pair (i,j) where i and j are
    dataset indices and where the known label of i and j are not the
    same.

    """
    def _simulate_must_not_link(self):
        while True:
            i,j = random.sample(range(self._N),2)
            if i > j: i,j = j,i
            if self._labels[i] != self._labels[j]: break
        return (i,j)

    """Simulate a list of pairwise must-link constraints

    """
    def _simulate_pairwise_constraints(self,count):
        random.seed(self._seed)
        constraints = set()
        while len(constraints) < count:
            constraints.add(self._simulate_must_link())
        return constraints

    """Convert a list of pairwise must-link constraints to a list of
    must-link sets.

    """
    def _pairwise_constraints_as_sets(self,constraints):
        return Graph.connected_components(range(self._N),constraints)

    """Partition the dataset into a list of must-link sets.  The 
    partition is mutually exclusive and exhaustive.
    
    """
    def simulate_constraints(self,count=10):
        constraints = self._simulate_pairwise_constraints(count=count)
        return self._pairwise_constraints_as_sets(constraints)

    """centers should be the 10 cluster centers from kmeans.py"""
    def save_centers_as_image(self,centers,filename,title=""):
        assert len(centers) == 10
        fig = pyplot.figure(figsize=(10,5))
        grid = ImageGrid(fig,111,nrows_ncols=(2,5),axes_pad=0.1)
        for ax,im in zip(grid,centers):
            im = im.reshape(self._dim)
            ax.imshow(im,cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title)
        pyplot.savefig(filename)

    def save_image(self,image,filename):
        fig = pyplot.figure()
        image = image.reshape(self._dim)
        pyplot.imshow(image,cmap="gray")
        pyplot.savefig(filename)

#from sortedcontainers import SortedSet as set
class Graph:

    @staticmethod
    def neighbor_map(edges):
        neighbors = defaultdict(set)
        for (i,j) in edges:
            neighbors[i].add(j)
            neighbors[j].add(i)
        return neighbors

    @staticmethod
    def connected_components(nodes,edges):
        #import pdb; pdb.set_trace()
        nodes = set(nodes)
        neighbors = Graph.neighbor_map(edges)
        ccs = list()
        while nodes:
            node = nodes.pop()
            component = set([node])
            open_list = set(neighbors[node])
            while open_list:
                node = open_list.pop()
                if node not in nodes: continue
                nodes.remove(node)
                component.add(node)
                open_list.update(neighbors[node])
            ccs.append(component)
        return ccs
