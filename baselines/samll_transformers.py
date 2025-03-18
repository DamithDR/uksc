from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import torch
import argparse
import os
import wandb

# Define label mappings
LABEL_MAP = {'dismiss': 0, 'allow': 1}
INV_LABEL_MAP = {0: 'dismiss', 1: 'allow'}


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
        eval_batch_size=8,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False
):
    # Initialize W&B run
    wandb.init(
        project="binary-classification",
        config={
            "model_type": model_type,
            "model_name": model_name,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "use_multiprocessing": use_multiprocessing,
            "use_multiprocessing_for_evaluation": use_multiprocessing_for_evaluation,
            "train_samples": len(train_data),
            "eval_samples": len(eval_data),
            "test_samples": len(test_data),
        }
    )

    # Define model arguments with W&B integration and multiprocessing settings
    args = {
        'num_train_epochs': num_train_epochs,
        'learning_rate': learning_rate,
        'train_batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size,
        'overwrite_output_dir': True,
        'evaluate_during_training': True,
        'evaluate_during_training_verbose': True,
        'use_cuda': True if torch.cuda.is_available() else False,
        'wandb_project': "binary-classification",
        'wandb_kwargs': {"name": f"{model_name}_{num_train_epochs}epochs"},
        'use_multiprocessing': use_multiprocessing,
        'use_multiprocessing_for_evaluation': use_multiprocessing_for_evaluation
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
    wandb.log({"validation_mcc": val_result['mcc'], "validation_loss": val_result['eval_loss']})

    # Evaluate on test set
    test_predictions, test_raw_outputs = model.predict(test_data['text'].tolist())
    # Calculate F1 scores for both macro and weighted averages
    test_f1_macro = f1_score(test_data['labels'], test_predictions, average='macro')
    test_f1_weighted = f1_score(test_data['labels'], test_predictions, average='weighted')

    # Map predictions back to string labels
    test_predictions_str = [INV_LABEL_MAP[pred] for pred in test_predictions]
    test_true_labels_str = [INV_LABEL_MAP[label] for label in test_data['labels']]

    # Prepare results string
    results = []
    results.append("Test Set Results:")
    results.append(f"F1 Score (macro): {test_f1_macro:.4f}")
    results.append(f"F1 Score (weighted): {test_f1_weighted:.4f}")
    results.append("\nDetailed Classification Report:")
    report = classification_report(test_data['labels'], test_predictions,
                                   target_names=['Dismiss', 'Allow'])
    results.append(report)

    # Print results to console
    print("\n".join(results))

    # Log test results to W&B
    wandb.log({"test_f1_macro": test_f1_macro, "test_f1_weighted": test_f1_weighted})
    wandb.log({"classification_report": wandb.Html("<pre>" + report + "</pre>")})

    # Save results to file
    safe_model_name = model_name.replace('/', '_')
    output_file = f"{safe_model_name}_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Epochs: {num_train_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Train Batch Size: {train_batch_size}\n")
        f.write(f"Eval Batch Size: {eval_batch_size}\n")
        f.write(f"Use Multiprocessing: {use_multiprocessing}\n")
        f.write(f"Use Multiprocessing for Evaluation: {use_multiprocessing_for_evaluation}\n")
        f.write("\nValidation Results:\n")
        f.write(str(val_result) + "\n")
        f.write("\n" + "\n".join(results))

    print(f"Results saved to {output_file}")

    # Finish W&B run
    wandb.finish()

    return model, test_predictions_str


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
    historic_df['labels'] = historic_df['labels'].map(LABEL_MAP)

    # Prepare test data
    test_df = test_df[['judgment_text', 'decision_label']].rename(
        columns={'judgment_text': 'text', 'decision_label': 'labels'}
    )
    test_df['labels'] = test_df['labels'].map(LABEL_MAP)

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
    parser.add_argument('--use_multiprocessing', type=bool, default=False,
                        help='Use multiprocessing for training (True/False)')
    parser.add_argument('--use_multiprocessing_for_evaluation', type=bool, default=False,
                        help='Use multiprocessing for evaluation (True/False)')

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
    model, test_predictions_str = train_binary_classifier(
        model_type=args.model_type,
        model_name=args.model_name,
        train_data=train_df,
        eval_data=eval_df,
        test_data=test_df,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        use_multiprocessing=args.use_multiprocessing,
        use_multiprocessing_for_evaluation=args.use_multiprocessing_for_evaluation
    )


if __name__ == "__main__":
    main()