from getdata import load, get
from rfc import avg_accuracy_rfc
from knn import avg_accuracy_knn
import time
load()

CLASSIFIER = avg_accuracy_knn # avg_accuracy_knn or avg_accuracy_rfc

# Since the labels in the dataset are in column 0, we need to shift them to the end for the classifier to work properly
data = get()
for i in range(len(data)):
    data[i] = data[i][1:] + [data[i][0]]

with open("shifted_data.csv", "w") as f:
    for row in data:
        f.write(",".join(row) + "\n")

