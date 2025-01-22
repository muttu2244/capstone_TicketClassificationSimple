from transformers import BertTokenizer, BertForSequenceClassification
import torch
from data_processing import preprocess_text, tokenize_data

'''
def predict_ticket(model_dir, ticket_text):
    """
    Predicts the category of a ticket using a fine-tuned BERT model.
    Args:
        model_dir (str): Path to the fine-tuned model.
        ticket_text (str): Text of the ticket.
    Returns:
        str: Predicted category.
    """
    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Preprocess and tokenize
    ticket_text = preprocess_text(ticket_text)
    inputs = tokenize_data([ticket_text], tokenizer)

    # Prediction
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, axis=-1).item()
    return predicted_label

if __name__ == "__main__":
    ticket = "Server down on production instance"
    category = predict_ticket("models/fine_tuned_bert", ticket)
    print(f"Predicted Category: {category}")

'''

def predict_multiple_tickets(tickets, model_dir):
    """
    Predicts categories for multiple tickets.

    Args:
        tickets (list of str): List of ticket texts to classify.
        model_dir (str): Path to the trained model.
    """
    #from transformers import BertTokenizer, BertForSequenceClassification
    #import torch

    # Define the mapping from numeric labels to category names
    label_map = {
        0: "Hardware",
        1: "Access",
        2: "Miscellaneous",
        3: "HR Support",
        4: "Purchase",
        5: "Administrative rights",
        6: "Storage",
        7: "Internal Project"
    }

    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Tokenize the tickets
    inputs = tokenizer(tickets, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Move inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, axis=-1)

    # Map numeric predictions to category names
    predicted_categories = [label_map[prediction.item()] for prediction in predictions]

    # Print results
    for ticket, category in zip(tickets, predicted_categories):
        print(f"Ticket: {ticket}")
        print(f"Predicted Category: {category}")
        print()

# Example usage
if __name__ == "__main__":
    tickets = [
        "Server down on production instance",
    "Need access to the internal project dashboard",
    "Purchase a new laptop for HR team",
    "Increase storage quota for the database",
    "Install new printer and monitors",
    "Deploy security patches for Windows servers",
    "Configure VPN access for new remote employees",
    "Email service experiencing delays",
    "Update firewall rules for new application",
    "Backup system failing periodic checks",
    "Replace faulty network switch in Building B",
    "Set up video conferencing system in Conference Room 3",
    "Upgrade RAM for design team workstations",
    "Fix broken SSO integration with third-party apps",
    "Cloud storage sync issues affecting multiple users"
    ]
    predict_multiple_tickets(tickets, "models/fine_tuned_bert")
