# Andrew Lai
Implementation of a TensorFlow convolutional neural network to classify images of 10 different types of clothing.

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

num_iterations = 5000 // this also causes a lot of variation in terms of training accuracy

# notes
Due to the validation split used (16%), the model will give a slightly different prediction every time. I tended to stick to 4 epochs for testing, which gave consistent accuracies of around 75%-85%, but the numbers predicted by the model itself were different every time due to the low number of iterations.

Uses https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb and https://github.com/rdcolema/tensorflow-image-classification as bases.
