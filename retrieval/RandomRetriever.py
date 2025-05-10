import pandas as pd


class RandomRetriever:

    def __init__(self, excel_file, sheet_name='data', text_column='background', label_column='decision_label',
                 reasoning_column='reasoning'):
        # Load data
        self.df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Store column names
        self.text_column = text_column
        self.label_column = label_column
        self.reasoning_column = reasoning_column

    def retrieve(self):
        idx = self.df.sample(n=1).index[0]
        results = []

        result = {
            'text': self.df[self.text_column].iloc[idx],
            'decision_label': self.df[self.label_column].iloc[idx],
            'reasoning': self.df[self.reasoning_column].iloc[idx],
        }
        results.append(result)

        return results
