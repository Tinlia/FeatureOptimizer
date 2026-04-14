import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from getdata import get

# LSVC Classifier. alpha isn't used in the current fitness function, but see the note at the end on its importance
def evaluate_lsvc(chromosome, alpha=None) -> float:
    data = get()
    X = np.array([row[:-1] for row in data], dtype=float) # NOTE: Assuming all features are numeric 
    y = np.array([row[-1] for row in data])               # NOTE: Assuming last column is the label

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    selected_indices = [i for i, gene in enumerate(chromosome) if gene == 1]

    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    clf = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)

    fitness = float(accuracy)
    # Currently, the fitness is just this accuracy
    # This can be updated to reflect importance of feature reduction by adding a term like:
    # fitness = alpha * accuracy + (1 - alpha) * (1 - (num_selected_features / total_features))
    
    return fitness