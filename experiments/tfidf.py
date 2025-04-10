import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load datasets
train_data = pd.read_excel('data/historic/historic_data.xlsx', sheet_name='data')
test_data = pd.read_excel('data/test_data_extended.xlsx', sheet_name='data')


# Convert labels to numeric for XGBoost (assuming 'allow'/'dismiss')
def encode_labels(labels):
    return labels.map({'dismiss': 0, 'allow': 1})


# Function to train, evaluate, and return macro F1 scores
def train_and_evaluate(X_train, X_test, y_train, y_test, experiment_name):
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Encode labels for XGBoost
    y_train_encoded = encode_labels(y_train)
    y_test_encoded = encode_labels(y_test)

    # Define models
    models = {
        'SVM': SVC(kernel='linear'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LR': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Store macro F1 scores
    macro_f1_scores = {}

    # Train and evaluate each model
    print(f"\nResults for {experiment_name}:")
    for model_name, model in models.items():
        # Train the model (use encoded labels for XGBoost, original for others)
        y_train_to_use = y_train_encoded if model_name == 'XGBoost' else y_train
        model.fit(X_train_tfidf, y_train_to_use)
        # Predict on test set
        y_pred = model.predict(X_test_tfidf)
        # For XGBoost, convert numeric predictions back to labels
        if model_name == 'XGBoost':
            y_pred = pd.Series(y_pred).map({0: 'dismiss', 1: 'allow'})
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Generate classification report
        report_dict = classification_report(y_test, y_pred, target_names=['dismiss', 'allow'], output_dict=True)
        report = classification_report(y_test, y_pred, target_names=['dismiss', 'allow'])
        macro_f1 = report_dict['macro avg']['f1-score']
        macro_f1_scores[model_name] = macro_f1

        # Print to console
        print(f"\n{model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print("Classification Report:")
        print(report)

        # Save to file
        filename = f"results/{experiment_name.lower()}_{model_name.lower().replace(' ', '_')}_results.txt"
        with open(filename, 'w') as f:
            f.write(f"{experiment_name} - {model_name}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
            f.write("Classification Report:\n")
            f.write(report)

    return macro_f1_scores


# Run experiments and collect macro F1 scores
print("Experiment 1: Predicting using Background")
X_train_bg = train_data['background'].fillna('')
X_test_bg = test_data['background'].fillna('')
y_train = train_data['decision_label']
y_test = test_data['decision_label']
bg_scores = train_and_evaluate(X_train_bg, X_test_bg, y_train, y_test, "Background")

print("\nExperiment 2: Predicting using Judgment")
X_train_judgment = train_data['judgment'].fillna('')
X_test_judgment = test_data['judgment'].fillna('')
y_train = train_data['decision_label']
y_test = test_data['decision_label']
judgment_scores = train_and_evaluate(X_train_judgment, X_test_judgment, y_train, y_test, "Judgment")

# Format and output the macro F1 table
table_output = (
    "method     SVM     KNN     LR     XGBoost\n"
    f"background {bg_scores['SVM']:.4f} {bg_scores['KNN']:.4f} {bg_scores['LR']:.4f} {bg_scores['XGBoost']:.4f}\n"
    f"judgment   {judgment_scores['SVM']:.4f} {judgment_scores['KNN']:.4f} {judgment_scores['LR']:.4f} {judgment_scores['XGBoost']:.4f}"
)
print("\nMacro F1 Scores Table:")
print(table_output)

# Save the table to a file
with open('results/macro_f1_scores.txt', 'w') as f:
    f.write("Macro F1 Scores Table:\n")
    f.write(table_output)