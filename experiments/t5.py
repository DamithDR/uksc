import logging
import os

import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def prepare_dataset(df, column):
    dataset = []
    for background, reason in zip(df[column], df['reasoning']):
        dataset.append(['reasoning', background, reason])

    return dataset


def run(column):
    df = pd.read_excel('data/historic/historic_data_with_reason.xlsx', sheet_name='data')

    train_data = prepare_dataset(df, column)

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["prefix", "input_text", "target_text"]
    train_df['target_text'] = train_df['target_text'].astype(str)

    train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=42)

    eval_df.columns = ["prefix", "input_text", "target_text"]
    eval_df['target_text'] = eval_df['target_text'].astype(str)

    # Configure the model
    model_args = T5Args()
    model_args.num_train_epochs = 200
    model_args.no_save = True
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.output_dir = 't5_outputs'

    model = T5Model("t5", "t5-base", args=model_args)

    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    result = model.eval_model(eval_df)

    test_df = pd.read_excel('data/test_data_extended.xlsx', sheet_name='data')
    to_predict = test_df[column]

    predictions = model.predict(to_predict)

    reasons_df = pd.DataFrame()
    reasons_df['gold'] = test_df['reasoning']
    reasons_df['predictions'] = predictions

    if not os.path.exists(f"outputs/t5_reasons_{column}.xlsx"):
        reasons_df.to_excel(f"outputs/t5_reasons_{column}.xlsx", sheet_name='data', index=False)
    else:
        with pd.ExcelWriter(f'outputs/t5_reasons_{column}.xlsx', mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            reasons_df.to_excel(writer, sheet_name='data', index=False)


if __name__ == '__main__':
    run('background')
    run('judgment')
