# Andrew Lai
**Slightly modified to run on an AWS Ubuntu Deep Learning AMI trained on an NVidia V100 16GB GPU, rather than my laptop's i7-5600u CPU; now reaches 100% accuracy on the training set, up to 96% accuracy on the validation split.

Implementation of a TensorFlow convolutional neural network to classify images of 10 different types of clothing.

Usage: just run prediction.py using python

As listed in prediction.py, here are the parameters:

# convolutional layer parameters
filter_size1 = 3 
num_filters1 = 32

filter_size2 = 3
num_filters2 = 32

filter_size3 = 3
num_filters3 = 64

fc_size = 128

num_channels = 3

img_size = 28

img_size_flat = img_size * img_size * num_channels

img_shape = (img_size, img_size)

classes = ['0','1','2','3','4','5','6','7','8','9']
num_classes = len(classes)

batch_size = 32

validation_size = .16 // this causes a lot of variation due to the changing validation splits

~~num_iterations = 5000

num_iterations = 58000

# notes
~~Due to the validation split used (16%), the model will give a slightly different prediction every time. I tended to stick to 4 epochs for testing, which gave consistent accuracies of around 75%-85%, but the numbers predicted by the model itself were different every time due to the low number of iterations.

With the updated number of iterations/epochs, model now gives 100% accuracy on the training set and averages 90% on the validation split, with an upper limit of 96%.

Uses https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb and https://github.com/rdcolema/tensorflow-image-classification as bases.
