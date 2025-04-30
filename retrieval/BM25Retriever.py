import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')


class BM25Retriever:
    def __init__(self, excel_file, sheet_name='data', text_column='background', label_column='decision_label',
                 reasoning_column='reasoning'):
        """
        Initialize BM25 Retriever with data from an Excel file.

        Args:
            excel_file (str): Path to the Excel file
            sheet_name (str): Name of the sheet containing data
            text_column (str): Column name containing text for retrieval
            label_column (str): Column name containing labels to retrieve
            reasoning_column (str): Column name containing reasoning to retrieve
        """
        # Load data
        self.df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Store column names
        self.text_column = text_column
        self.label_column = label_column
        self.reasoning_column = reasoning_column

        # Preprocess texts
        self.stop_words = set(stopwords.words('english'))
        self.corpus = self.df[text_column].astype(str).tolist()
        self.tokenized_corpus = [self._preprocess(text) for text in self.corpus]

        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _preprocess(self, text):
        """
        Preprocess text by tokenizing, removing stopwords, and punctuation.

        Args:
            text (str): Input text

        Returns:
            list: Processed tokens
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())

        # Remove punctuation and stopwords
        tokens = [token for token in tokens
                  if token not in string.punctuation
                  and token not in self.stop_words]

        return tokens

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k relevant documents for a given query.

        Args:
            query (str): Search query
            top_k (int): Number of documents to retrieve

        Returns:
            list: List of dictionaries containing retrieved documents and scores
        """
        # Preprocess query
        tokenized_query = self._preprocess(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = scores.argsort()[-top_k:][::-1]

        # Prepare results
        results = []
        for idx in top_k_indices:
            result = {
                'text': self.corpus[idx],
                'decision_label': self.df[self.label_column].iloc[idx],
                'reasoning': self.df[self.reasoning_column].iloc[idx],
                'score': scores[idx]
            }
            results.append(result)

        return results


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = BM25Retriever(
        excel_file="data/historic/historic_data_with_reason.xlsx",
        sheet_name="data",
        text_column="background",
        label_column="decision_label",
        reasoning_column="reasoning"
    )

    # Example query
    query = "example query text"
    results = retriever.retrieve(query, top_k=5)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:100]}...")  # First 100 chars
        print(f"Decision Label: {result['decision_label']}")
        print(f"Reasoning: {result['reasoning'][:100]}...")  # First 100 chars