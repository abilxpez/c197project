import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support
from typing import List
from skmultilearn.model_selection import iterative_train_test_split

import json
import numpy as np

def load_data(labeled_json_path, unlabeled_npz_path):
    # Load labeled speeches (with embeddings and labels)
    with open(labeled_json_path, 'r') as f:
        labeled_data = json.load(f)

    # Load unlabeled embeddings from .npz
    npz_data = np.load(unlabeled_npz_path, allow_pickle=True)

    unlabeled_data = [
        {
            "id": int(doc_id),
            "date": date,
            "embedding": embedding
        }
        for doc_id, date, embedding in zip(
            npz_data["doc_ids"],
            npz_data["dates"],
            npz_data["embeddings"]
        )
    ]

    return labeled_data, unlabeled_data

def extract_features_and_labels(labeled_data):
    X = np.array([item['embedding'] for item in labeled_data])
    y_raw = [item['labels'] for item in labeled_data]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_raw)
    return X, Y, mlb

def iterative_split(X, Y, test_size=0.2):

    X = X.astype(np.float32)
    Y = Y.astype(np.int32)

    X_train, Y_train, X_test, Y_test = iterative_train_test_split(
        X, Y, test_size=test_size
    )
    return X_train, X_test, Y_train, Y_test


def train_classifier(X_train, Y_train):
    base_model = LogisticRegression(max_iter=5000, class_weight='balanced')
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    model = OneVsRestClassifier(calibrated_model)    
    model.fit(X_train, Y_train)
    return model

def evaluate(model, X_test, Y_test, mlb):
    Y_pred = model.predict(X_test)
    print("Evaluation on test set:\n")
    print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))

def predict_labels_per_class_threshold(
    model,
    unlabeled_data: List[dict],
    mlb: MultiLabelBinarizer,
    thresholds: np.ndarray
) -> List[dict]:
    """
    Predict labels using different thresholds for each class.
    """
    embeddings = np.array([item["embedding"] for item in unlabeled_data])
    probas = model.predict_proba(embeddings)

    # Apply per-class thresholds
    predictions = (probas >= thresholds).astype(int)

    for item, label_vec in zip(unlabeled_data, predictions):
        item["labels"] = list(mlb.inverse_transform(np.array([label_vec]))[0])
    return unlabeled_data

def save_predictions(path, data):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    # Recursively convert all values in the data
    converted_data = [
        {k: convert(v) for k, v in item.items()}
        for item in data
    ]

    with open(path, "w") as f:
        json.dump(converted_data, f, indent=2)

def tune_per_class_thresholds(model, X_val, Y_val, thresholds=np.arange(0.1, 0.95, 0.05)):
    """
    Tune threshold per label using validation set to maximize F1 score.
    """
    probas = model.predict_proba(X_val)
    n_classes = Y_val.shape[1]
    best_thresholds = []

    for i in range(n_classes):
        best_f1 = 0
        best_thresh = 0.5  # default
        for t in thresholds:
            preds = (probas[:, i] >= t).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                Y_val[:, i], preds, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        best_thresholds.append(best_thresh)

    return np.array(best_thresholds)

def main():
    # File paths
    labeled_path = r'congress_sampled_labeled_speeches_embedded_10k.json'
    unlabeled_path = r'congress_speech_embeddings_99k_filtered.npz'
    output_path = 'newly_labeled_speeches.json'

    # Load data
    labeled_data, unlabeled_data = load_data(labeled_path, unlabeled_path)

    # Prepare features and labels
    X, Y, mlb = extract_features_and_labels(labeled_data)

    # Train/test split
    X_train, X_test, Y_train, Y_test = iterative_split(X, Y, test_size=0.2)

    # Train model
    model = train_classifier(X_train, Y_train)

    # Evaluate model
    evaluate(model, X_test, Y_test, mlb)

    # Tune per-class thresholds
    print("Tuning per-class thresholds to maximize F1...")
    per_class_thresholds = tune_per_class_thresholds(model, X_test, Y_test)

    # Predict with tuned thresholds
    labeled_unlabeled_data = predict_labels_per_class_threshold(model, unlabeled_data, mlb, per_class_thresholds)

    # Save output
    save_predictions(output_path, labeled_unlabeled_data)

    print(f"Done! Labeled data saved to {output_path}")

if __name__ == '__main__':
    main()