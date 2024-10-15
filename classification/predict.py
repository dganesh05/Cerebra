import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('./results')
model = BertForSequenceClassification.from_pretrained('./results')

model.eval()

def predict_message(message):
    inputs = tokenizer(message, padding=True, truncation=True, return_tensors='pt', max_length=512)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)

    predicted_label = torch.argmax(probabilities, dim=1).item()

    return predicted_label
