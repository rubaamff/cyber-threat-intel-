import time
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Assuming these utility functions are available or imported from evaluation_metrics.py and blockchain_logger.py
from evaluation_metrics import save_conf_matrix, save_class_report
from blockchain_logger import log_predictions

def train_supervised_models(X_train, X_test, y_train, y_test, blockchain_path):
    models = {
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}, Training Time: {elapsed:.2f}s")

        joblib.dump(model, f"models/{name}.joblib")
        save_conf_matrix(y_test, y_pred, name)
        save_class_report(y_test, y_pred, name)
        log_predictions(model_name=name, y_pred=y_pred, y_true=y_test, blockchain_path=blockchain_path)

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

    return models

def train_unsupervised_models(X_test, y_test, blockchain_path):
    unsupervised = {
        "KMeans": KMeans(n_clusters=2, n_init=10, random_state=42),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "IsolationForest": IsolationForest(random_state=42, contamination=0.1)
    }

    for name, model in unsupervised.items():
        start = time.time()
        if name == "DBSCAN":
            pred = model.fit_predict(X_test)
            pred = np.where(pred == -1, 1, 0)
        else:
            pred = model.fit_predict(X_test)
            pred = np.where(pred == -1, 1, 0) if name == "IsolationForest" else pred

        elapsed = time.time() - start

        acc = accuracy_score(y_test, pred)
        print(f"{name} Accuracy: {acc:.4f}, Training Time: {elapsed:.2f}s")

        save_conf_matrix(y_test, pred, name)
        save_class_report(y_test, pred, name)
        log_predictions(model_name=name, y_pred=pred, y_true=y_test, blockchain_path=blockchain_path)

def train_lstm_model(X_train, X_test, y_train, y_test, blockchain_path):
    X_lstm_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_lstm_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm_model = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1])),
        Dense(32, activation=\'relu\'),
        Dense(1, activation=\'sigmoid\')
    ])

    lstm_model.compile(loss=\'binary_crossentropy\', optimizer=\'adam\', metrics=[\'accuracy\'])

    start = time.time()
    history = lstm_model.fit(X_lstm_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    elapsed = time.time() - start

    lstm_pred = (lstm_model.predict(X_lstm_test) > 0.5).astype(int).flatten()
    print(f"LSTM Accuracy: {accuracy_score(y_test, lstm_pred):.4f}, Training Time: {elapsed:.2f}s")

    lstm_model.save("models/LSTM_model.h5")
    save_conf_matrix(y_test, lstm_pred, "LSTM")
    save_class_report(y_test, lstm_pred, "LSTM")
    log_predictions(model_name="LSTM", y_pred=lstm_pred, y_true=y_test, blockchain_path=blockchain_path)

    fpr, tpr, _ = roc_curve(y_test, lstm_model.predict(X_lstm_test).flatten())
    plt.plot(fpr, tpr, label=f"LSTM (AUC={auc(fpr,tpr):.2f})")

def train_ensemble_model(X_train, X_test, y_train, y_test, supervised_models, blockchain_path):
    ensemble = VotingClassifier(
        estimators=[
            (\'nb\', supervised_models["NaiveBayes"]),
            (\'svm\', supervised_models["SVM"]),
            (\'rf\', supervised_models["RandomForest"])
        ],
        voting=\'soft\'
    )

    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ens):.4f}")

    joblib.dump(ensemble, "models/Ensemble.joblib")
    save_conf_matrix(y_test, y_pred_ens, "Ensemble")
    save_class_report(y_test, y_pred_ens, "Ensemble")
    log_predictions(model_name="Ensemble", y_pred=y_pred_ens, y_true=y_test, blockchain_path=blockchain_path)

    fpr, tpr, _ = roc_curve(y_test, ensemble.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"Ensemble (AUC={auc(fpr,tpr):.2f})")

    plt.plot([0, 1], [0, 1], \'k--\', label=\'Random\')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/roc_curves.png", dpi=300, bbox_inches=\'tight\')
    plt.show()
    plt.close()

if __name__ == '__main__':
    # This script is intended to be imported and used by main_notebook.ipynb
    pass


