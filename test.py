from getdata import load, get
from rfc import avg_accuracy_rfc
from knn import avg_accuracy_knn
import time
load()

CLASSIFIER = avg_accuracy_knn # avg_accuracy_knn or avg_accuracy_rfc

start = time.time()
print(f"{CLASSIFIER((1, 1, 1, 1, 1, 1, 1, 1), runs=10)*100:.2f}%") # THIS IS HUGE
print(f"Classifier used: {CLASSIFIER.__name__}")
end = time.time()
print(f"8ft Time taken: {end - start} seconds\n")

start = time.time()
print(f"{CLASSIFIER((0, 1, 0, 0, 0, 1, 0, 0), runs=10)*100:.2f}%") # THIS IS HUGE
end = time.time()
print(f"2ft Time taken: {end - start} seconds\n")