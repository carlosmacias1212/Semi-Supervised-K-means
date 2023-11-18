# CS4412 : Data Mining
# Kennesaw State University

"""In this project, we will explore the MNIST handwritten digits
dataset, available at:
  http://yann.lecun.com/exdb/mnist/

The dataset included in this project was downloaded using TensorFlow:
  https://www.tensorflow.org/datasets/catalog/mnist
"""

from kmeans import KMeans # k-means included with project
from mnist import Mnist   # data structure for the MNIST dataset

print("= reading dataset ...")
dataset = Mnist()
# the following command shrinks the dataset by half
# comment the following out to use the full dataset
dataset.shrink(factor=0.5)

# the MNIST dataset has 55,000 images
# each image has 28x28 = 784 grayscale pixels
images = dataset.images()

# view an example image from the dataset (the 100th image in the dataset)
filename = "output/digit-example.png"
dataset.save_image(images[100],filename)

print("= running k-means (without constraints) ...")
# try a new RNG seed if k-means doesn't converge to a good optima
kmeans = KMeans(images,num_clusters=10,max_iters=10,seed=0)
centers = kmeans.cluster()

# save centers to file as an image
filename = "output/centers.png"
print("= saving centers as %s ..." % filename)
dataset.save_centers_as_image(centers,filename,"k-means")

# simulate constraints
constraints = dataset.simulate_constraints(20000)

# try a new RNG seed if k-means doesn't converge to a good optima
print("= running k-means (with constraints) ...")
kmeans = KMeans(images,num_clusters=10,max_iters=10,seed=0,
                constraints=constraints)
centers = kmeans.cluster()

# save centers to file as a separate image
filename = "output/centers-constraints.png"
print("= saving centers as %s ..." % filename)
dataset.save_centers_as_image(centers,filename,"constrained k-means")
