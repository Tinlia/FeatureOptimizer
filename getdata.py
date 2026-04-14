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

def apply_mask(c: tuple[int]) -> None:
    # Save the dataset with only the selected features (for testing purposes)
    with open("output/reduced_data.csv", "w") as f:
        header = [f"feature_{i}" for i in range(len(c)) if c[i] == 1] + ["label"]
        f.write(",".join(header) + "\n")
        for row in dataset:
            selected_features = [row[i] for i in range(len(c)) if c[i] == 1]
            f.write(",".join(selected_features + [row[-1]]) + "\n")