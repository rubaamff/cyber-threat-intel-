# Cyber Threat Intelligence Framework

This repository contains the source code, processed datasets, and experimental scripts used in our cyber threat intelligence study. It includes a reproducible Jupyter notebook and detailed documentation to ensure full transparency and facilitate independent replication.

## Project Structure

- `README.md`: This file.
- `requirements.txt`: Python dependencies required to run the project.
- `preprocess_data.py`: Script for data preprocessing and feature engineering.
- `extract_iocs.py`: Script for extracting Indicators of Compromise (IOCs).
- `train_classifiers.py`: Script for training and evaluating machine learning models.
- `blockchain_logger.py`: Module for logging predictions to a simulated blockchain.
- `evaluation_metrics.py`: Module for calculating and reporting evaluation metrics.
- `main_notebook.ipynb`: Jupyter notebook demonstrating the full pipeline.
- `sample_data/`: Directory containing sample datasets.
  - `cicids_subset.csv`: A subset of the CICIDS2017 dataset.
  - `annotated_threats.json`: Sample file for annotated threats (placeholder).
- `models/`: Directory for saving trained models and related outputs.
  - `saved_model_weights/`: Directory for saving model weights (e.g., for LSTM).

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone [GitHub/Zenodo DOI link]
    cd cyber-threat-intel-framework
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter Notebook:**
    Open `main_notebook.ipynb` in a Jupyter environment to reproduce the experiments and analysis.

## Data Sources

This study utilizes subsets of the NSL-KDD and CICIDS2017 datasets for training and evaluation of the models.

## Models

The framework includes implementations of various machine learning models, including Naive Bayes, SVM, Random Forest, KMeans, DBSCAN, Isolation Forest, and LSTM, along with an ensemble model.

## Contributions

Contributions are welcome. Please refer to the contribution guidelines for more information.

## License

[License Information]

## Contact

[Contact Information]

