import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

def save_conf_matrix(y_true, y_pred, title):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=\'d\', cmap=\'Blues\')
    plt.title(f\"{title} Confusion Matrix\")
    plt.xlabel(\"Predicted\")
    plt.ylabel(\"Actual\")
    plt.tight_layout()
    plt.savefig(f\"models/{title}_cm.png\")
    plt.close()

def save_class_report(y_true, y_pred, name):
    """
    Generates and saves a classification report as a JSON file.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f\"models/{name}_report.json\", \"w\") as f:
        json.dump(report, f, indent=2)

if __name__ == \'__main__\':
    # Example usage:
    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import train_test_split
    # X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # y_pred = [0 if val < 0.5 else 1 for val in X_test[:, 0]] # Dummy predictions

    # save_conf_matrix(y_test, y_pred, "TestModel")
    # save_class_report(y_test, y_pred, "TestModel")
    pass


