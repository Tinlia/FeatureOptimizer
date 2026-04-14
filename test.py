from getdata import load, get
from rfc import avg_accuracy_rfc
from knn import avg_accuracy_knn
import time
load()

CLASSIFIER = avg_accuracy_knn # avg_accuracy_knn or avg_accuracy_rfc

# Since the labels in the dataset are in column 0, we need to shift them to the end for the classifier to work properly
# We cant use the get() method since it gets rid of the header


