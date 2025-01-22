from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from data_processing import load_data, preprocess_text

def evaluate_model(model_dir, test_file, batch_size=16):
    """
    Evaluates a fine-tuned BERT model with batch processing.
    Args:
        model_dir (str): Path to the fine-tuned model.
        test_file (str): Path to the test dataset.
        batch_size (int): Batch size for evaluation.
    """
    # Load model and tokenizer
    print("[INFO] Loading model and tokenizer...")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.eval()  # Set model to evaluation mode
    print("[INFO] Model and tokenizer loaded.")

    # Load and preprocess data
    print("[INFO] Loading and preprocessing data...")
    data = load_data(test_file)
    data["text"] = data["text"].apply(preprocess_text)
    labels = data["label"].astype("category").cat.codes
    print(f"[INFO] Loaded {len(data)} samples.")

    # Tokenize data
    print("[INFO] Tokenizing data...")
    inputs = tokenizer(
        data["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    print("[INFO] Data tokenized.")

    # Create dataset and DataLoader
    dataset = TensorDataset(
        inputs["input_ids"],
        inputs["attention_mask"],
        torch.tensor(labels.values)
    )
    data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    # Model evaluation
    print("[INFO] Starting model evaluation in batches...")
    predictions = []
    true_labels = []
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, axis=-1).cpu().numpy()
            predictions.extend(batch_predictions)
            true_labels.extend(labels.cpu().numpy())
            print(f"[INFO] Processed batch {step + 1}/{len(data_loader)}")

    # Generate classification report
    print("[INFO] Evaluation completed. Generating report...")
    report = classification_report(true_labels, predictions, target_names=list(data["label"].unique()))
    print(report)

if __name__ == "__main__":
    # Adjust the batch size if needed
    evaluate_model("models/fine_tuned_bert", "data/raw/all_tickets_processed_improved_v3.csv", batch_size=16)
