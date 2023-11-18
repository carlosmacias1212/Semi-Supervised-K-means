# CS4412 : Data Mining
# Kennesaw State University

"""We provide a basic implementation of the k-means clustering
algorithm here."""

import numpy         # numpy is a library for matrix math
import random        # random number generator
from math import inf # inf is a floating point number for infinity

class KMeans:

    def __init__(self,data,num_clusters=10,max_iters=100,seed=0,
                 constraints=None):
        self.data = data
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.seed = seed
        self.count = 0

        self._N = len(data)       # size of dataset
        self._dim = len(data[0])  # length of each example
        if constraints is None:
            # generate vacuous constraints, where each constraint set
            # is composed of a single dataset index (i.e., each index
            # is unconstrained)
            self.constraints = [ set([i]) for i in range(self._N) ]
        else:
            self.constraints = constraints

        # initialize
        random.seed(self.seed)    # seed the random number generator
        self._seed_centers()      # init cluster centers randomly
        self._iterations = 0      # iteration counter

    """choose k constraint sets to use as initial centers"""
    def _k_sets(self):
        # MODIFY THIS FUNCTION
        # use the first k constraint sets as cluster centers
        # Modification: sort sets by length (desc) and return
        self.constraints_largest = sorted(self.constraints, key=lambda s: len(s), reverse=True)

        return self.constraints_largest[:self.num_clusters]

    """given a constraint set, return a summary "instance" to use to 
    pick the center closest to this set"""
    def _summarize_set(self,constraint):
        # MODIFY THIS FUNCTION
        # fetch the set of data points from the given set of data indices
        instances = [ self.data[inst] for inst in constraint ]

        mean = numpy.mean(instances, axis=0)

        # return the mean of
        return mean

    """initialize the centers of each cluster, randomly"""
    def _seed_centers(self):
        assert self.num_clusters <= len(self.constraints)
        # initialize labels and centers
        self._labels = numpy.zeros(self._N,dtype=int)
        self._centers = numpy.zeros([self.num_clusters,self._dim])
        # pick k sets and use their means as initial centers
        for i,constraint in enumerate(self._k_sets()):
            instances = [ self.data[inst] for inst in constraint ]
            self._centers[i] = numpy.mean(instances,axis=0)


    """given instance, return the index to the closest cluster"""
    def _closest_center(self,instance):
        # compute difference
        distances = self._centers - instance
        # square the difference (slower, so commented out)
        # distances = [ numpy.dot(d,d) for d in distances ]
        # square the difference (faster)
        distances = numpy.einsum('ij,ij->i',distances,distances)
        return numpy.argmin(distances) # index of the closest cluster

    """re-assign each data instance to a new cluster"""
    def _e_step(self):
        for constraint in self.constraints:
            summary = self._summarize_set(constraint)
            closest = self._closest_center(summary)
            for i in constraint:
                self._labels[i] = closest

    """based on current assignments, re-estimate the centers"""
    def _m_step(self):
        # initialize
        self._centers = numpy.zeros([self.num_clusters,self._dim])
        self._counts = numpy.zeros(self.num_clusters)
        # compute the averages (sum and normalize)
        for instance,label in zip(self.data,self._labels):
            self._centers[label] += instance
            self._counts[label] += 1
        self._counts[self._counts == 0] = 1 # for empty clusters
        for i in range(self.num_clusters):
            self._centers[i] /= self._counts[i]

    """run a few iterations of k-means clustering"""
    def cluster(self):
        max_iters = self._iterations + self.max_iters
        for _ in range(self.max_iters):
            self._iterations += 1
            print("iteration (%d/%d)" % (self._iterations,max_iters))
            self._e_step()
            self._m_step()
        return self._centers
