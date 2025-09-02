import json
from datetime import datetime

def log_predictions(model_name, y_pred, y_true, blockchain_path="models/blockchain.json"):
    """
    Logs model predictions and actual values to a JSON file simulating a blockchain.
    Each entry includes model name, timestamp, index, prediction, actual value, and a hash.
    """
    with open(blockchain_path, "r+") as f:
        chain = json.load(f)
        for i, (pred, actual) in enumerate(zip(y_pred, y_true)):
            entry = {
                "model": model_name,
                "timestamp": str(datetime.utcnow()),
                "index": len(chain) + i,
                "prediction": int(pred),
                "actual": int(actual),
                "hash": hash(f"{model_name}{pred}{actual}{len(chain)+i}") # Simple hash for demonstration
            }
            chain.append(entry)
        f.seek(0)
        json.dump(chain, f, indent=2)

if __name__ == '__main__':
    # Example usage:
    # To initialize an empty blockchain.json if it doesn't exist
    # import os
    # if not os.path.exists("models/blockchain.json"):
    #     with open("models/blockchain.json", "w") as f:
    #         json.dump([], f)

    # log_predictions("TestModel", [0, 1, 0], [0, 1, 1])
    # with open("models/blockchain.json", "r") as f:
    #     print(json.load(f))
    pass


