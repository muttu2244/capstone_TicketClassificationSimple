import pandas as pd
from transformers import BertTokenizer

def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def preprocess_text(text):
    """
    Cleans text by removing extra spaces and punctuation.
    Args:
        text (str): Input text.
    Returns:
        str: Cleaned text.
    """
    return text.strip().lower()

def tokenize_data(data, tokenizer, max_length=128):
    """
    Tokenizes the text data using the BERT tokenizer.
    Args:
        data (list): List of text samples.
        tokenizer (BertTokenizer): BERT tokenizer.
        max_length (int): Maximum sequence length.
    Returns:
        dict: Tokenized data.
    """
    return tokenizer(
        data,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
