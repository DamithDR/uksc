from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import torch
import argparse
import os


# Function to create and train the model
def train_binary_classifier(
        model_type,
        model_name,
        train_data,
        eval_data,
        test_data,
        num_train_epochs=3,
        learning_rate=2e-5,
        train_batch_size=8,
        eval_batch_size=8
):
    # Define model arguments
    args = {
        'num_train_epochs': num_train_epochs,
        'learning_rate': learning_rate,
        'train_batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size,
        'overwrite_output_dir': True,
        'evaluate_during_training': True,
        'evaluate_during_training_verbose': True,
        'use_cuda': True if torch.cuda.is_available() else False,
    }

    # Initialize the model
    model = ClassificationModel(
        model_type=model_type,
        model_name=model_name,
        num_labels=2,
        args=args
    )

    # Train the model
    model.train_model(
        train_df=train_data,
        eval_df=eval_data
    )

    # Evaluate on validation set during training
    val_result, _, _ = model.eval_model(eval_data)
    print("Validation results:", val_result)

    # Evaluate on test set
    test_predictions, test_raw_outputs = model.predict(test_data['text'].tolist())
    test_f1 = f1_score(test_data['labels'], test_predictions, average='weighted')

    # Prepare results string
    results = []
    results.append("Test Set Results:")
    results.append(f"F1 Score (weighted): {test_f1:.4f}")
    results.append("\nDetailed Classification Report:")
    report = classification_report(test_data['labels'], test_predictions,
                                   target_names=['Dismiss', 'Allow'])
    results.append(report)

    # Print results to console
    print("\n".join(results))

    # Save results to file (replace / with _ in model_name for valid filename)
    safe_model_name = model_name.replace('/', '_')
    output_file = f"{safe_model_name}_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Epochs: {num_train_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Train Batch Size: {train_batch_size}\n")
        f.write(f"Eval Batch Size: {eval_batch_size}\n")
        f.write("\nValidation Results:\n")
        f.write(str(val_result) + "\n")
        f.write("\n" + "\n".join(results))

    print(f"Results saved to {output_file}")

    return model, test_predictions


# Load and prepare data
def load_and_prepare_data():
    # Load historic data
    historic_path = "data/historic/historic_data.xlsx"
    historic_df = pd.read_excel(historic_path)

    # Load test data
    test_path = "data/UKSC_dataset_extended.xlsx"
    test_df = pd.read_excel(test_path)

    # Prepare historic data
    historic_df = historic_df[['judgment', 'decision_label']].rename(
        columns={'judgment': 'text', 'decision_label': 'labels'}
    )

    # Prepare test data
    test_df = test_df[['judgment_text', 'decision_label']].rename(
        columns={'judgment_text': 'text', 'decision_label': 'labels'}
    )

    # Split historic data into train (90%) and validation (10%)
    train_df, eval_df = train_test_split(historic_df, test_size=0.1, random_state=42)

    return train_df, eval_df, test_df


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a binary classifier with SimpleTransformers")
    parser.add_argument('--model_type', type=str, default='bert',
                        help='Model type (e.g., bert, roberta, distilbert)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Model name (e.g., bert-base-uncased, roberta-base)')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Evaluation batch size')

    return parser.parse_args()


# Main execution
def main():
    # Parse command-line arguments
    args = parse_args()

    # Load data
    train_df, eval_df, test_df = load_and_prepare_data()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(eval_df)}")
    print(f"Test samples: {len(test_df)}")

    # Train the model using command-line arguments
    model, test_predictions = train_binary_classifier(
        model_type=args.model_type,
        model_name=args.model_name,
        train_data=train_df,
        eval_data=eval_df,
        test_data=test_df,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size
    )


if __name__ == "__main__":
    main()