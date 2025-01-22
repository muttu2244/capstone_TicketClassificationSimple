from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example preprocessing
texts = ["The server is down. Restart needed.", "Please create a new user account."]
categories = [0, 1]  # Map categories (Incident -> 0, Service Request -> 1)

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
