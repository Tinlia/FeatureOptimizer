# Dataset Loader
dataset = []

def load():
    global dataset
    with open("data/data.csv", "r") as f:
        next(f) # Skip header
        for line in f:
            dataset.append(line.strip().split(","))

def get() -> list[list]:
    return dataset

def feature_count() -> int:
    return len(dataset[0]) - 1 # Subtract 1 for label column
