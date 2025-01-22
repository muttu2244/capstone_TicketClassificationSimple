from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from data_processing import load_data, preprocess_text, tokenize_data
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

def train_model(train_file, output_dir="models/fine_tuned_bert"):
    """
    Fine-tunes a BERT model for ticket classification.
    Args:
        train_file (str): Path to the training dataset.
        output_dir (str): Path to save the fine-tuned model.
    """
    # Load and preprocess data
    data = load_data(train_file)
    data["text"] = data["text"].apply(preprocess_text)
    labels = data["label"].astype("category").cat.codes
    
    # Split data into train and evaluation sets
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        data["text"], labels, test_size=0.2, random_state=42
    )
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    eval_dataset = Dataset.from_dict({"text": eval_texts, "label": eval_labels})

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=len(labels.unique())
    )

    # Tokenize data
    def tokenize_fn(example):
        return tokenizer(
            example["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Updated from evaluation_strategy to eval_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        save_strategy="epoch"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Added evaluation dataset
        # Removed deprecated tokenizer parameter
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    #train_model("data/sample_data.csv") 
    train_model("data/raw/all_tickets_processed_improved_v3.csv")