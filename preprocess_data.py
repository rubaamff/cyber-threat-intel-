# Data preprocessing and feature engineering script

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

def load_and_preprocess_data(nsl_path, cic_path, sample_size=50000, random_state=42):
    """
    Loads NSL-KDD and CICIDS2017 datasets, preprocesses them, and merges them.
    """
    # Load NSL-KDD dataset
    nsl = pd.read_csv(nsl_path, header=None)
    nsl.columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'label', 'difficulty'
    ]
    nsl = nsl.sample(n=sample_size, random_state=random_state)
    nsl['label'] = nsl['label'].apply(lambda x: 0 if x == 'normal' else 1)
    nsl = nsl.select_dtypes(include=[np.number])

    # Load CICIDS2017 dataset
    cic = pd.read_csv(cic_path).sample(n=sample_size, random_state=random_state)
    cic.columns = cic.columns.str.strip()
    cic = cic.replace([np.inf, -np.inf], np.nan).dropna()
    cic['label'] = (cic['Label'] != 'BENIGN').astype(int)
    cic = cic.select_dtypes(include=[np.number])

    # Feature mapping between datasets
    mapping = {
        'src_bytes': 'Flow Bytes/s', 'dst_bytes': 'Flow Packets/s',
        'wrong_fragment': 'Fwd Header Length', 'num_failed_logins': 'Fwd Packets/s',
        'hot': 'Flow IAT Mean', 'logged_in': 'Flow IAT Max', 'count': 'Fwd IAT Total',
        'srv_count': 'Subflow Fwd Bytes', 'same_srv_rate': 'Fwd IAT Std',
        'dst_host_srv_count': 'Flow IAT Std', 'dst_host_same_srv_rate': 'Idle Std',
        'dst_host_diff_srv_rate': 'Idle Max',
        'dst_host_serror_rate': 'Fwd Header Length.1',
        'dst_host_srv_serror_rate': 'Flow Duration', 'srv_rerror_rate': 'Init_Win_bytes_forward',
        'srv_serror_rate': 'Bwd Packet Length Min', 'rerror_rate': 'Bwd Packet Length Max',
        'num_compromised': 'Bwd Packet Length Mean'
    }

    # Select and rename features for both datasets
    nsl2 = nsl[list(mapping.keys()) + ['label']].copy()
    cic2 = cic[[mapping[k] for k in mapping if mapping[k] in cic.columns] + ['label']].copy()
    cic2.columns = list(mapping.keys())[:len(cic2.columns)-1] + ['label']

    # Merge datasets
    df = pd.concat([nsl2, cic2], ignore_index=True).dropna()

    return df

def feature_selection_and_scaling(df, mi_threshold=0.01, random_state=42):
    """
    Performs feature selection using mutual information and scales the data.
    """
    X = df.drop("label", axis=1)
    y = df["label"]

    mi = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    top_features = X.columns[mi > mi_threshold]
    X = X[top_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, top_features

if __name__ == '__main__':
    # Example usage (assuming data files are available)
    # df = load_and_preprocess_data('/content/KDDTrain+.txt', '/content/cicids2017/MachineLearningCSV/Wednesday-workingHours.pcap_ISCX.csv')
    # X_train, X_test, y_train, y_test, scaler, top_features = feature_selection_and_scaling(df)
    # print("Data preprocessing complete.")
    pass
    # This script is intended to be imported and used by main_notebook.ipynb or train_classifiers.py
    pass


