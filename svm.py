'''
	A SVM image classifier for X-ray images

	The goal of this program is to predict the correct class of an 
	x-ray image. The available classes are:
	1. frontal wrist, right
	2. frontal wrist, left
	3. side wrist, right
	4. side wrist, left
  5. front wrists
  6. side wrists

	Code inspitest_ration from this example:
	http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.ndimage import imread
from sklearn import datasets, svm, metrics


# Set path variables
path = "../Images/32x32/"
classes = [1,2,3,4]


# Create wrists dataset and matching target array
wrists = []
target = []


# Load images and their class into wrists and target 
for c in classes:
  for file in os.listdir(path + str(c)):
    wrists.append(imread(path + str(c) + "/" + str(file), True))
    target.append(c)
print("Size of training data: ", len(wrists))


# Shuffle wrists and target
wrists_shuf = []
target_shuf = []
index_shuf = list(range(len(wrists)))
shuffle(index_shuf)
for i in index_shuf:
    wrists_shuf.append(wrists[i])
    target_shuf.append(target[i])


# Final arrays
wrists = np.asarray(wrists_shuf)
target = np.asarray(target_shuf)


# Create test data
test = []

# Load test images into array
for file in os.listdir(path + "test/"):
  test.append(imread(path + "test/" + str(file), True))
# print("Size of test data: ", len(test))


# Flatten the image data and create a (sample, feature) matrix
n_samples = len(wrists)
data = wrists.reshape((n_samples, -1))


# Create a classifier: a support vector classifier
classifier = svm.SVC(kernel="poly", degree=6)


# Using 50% of of data for training data
test_ratio = 0.2
print("Size of test data: ", 0.2 * len(data))
classifier.fit(data[:test_ratio*n_samples], target[:test_ratio*n_samples])


# Now predict the value of the digit on the second half:
expected = target[test_ratio*n_samples:]
predicted = classifier.predict(data[test_ratio*n_samples:])

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))