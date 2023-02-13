import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AdamW
from transformers.models import bert
from datasets import DatasetDict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def read_dataset(file_path:str, tokenizer:AutoTokenizer):
    dataset = pd.read_csv(file_path)
    df = dataset.iloc[1:50000,:]
    for i in range(len(df.values)):
        df.values[i, 1] =  1 if df.values[i, 1] == 'positive' else 0
    x_encodings = tokenizer(list(df.values[:,0]), truncation=True, padding ='max_length')
    data = {
        "input_ids": x_encodings.input_ids,
        "attention_mask":x_encodings.attention_mask,
        "labels":df.values[:,1]
            }
    return data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, index):
        return {key: torch.tensor(val[index]) for key, val in self.data.items()}
    
torch.cuda.empty_cache()
data = read_dataset('IMDB_Dataset.csv', AutoTokenizer.from_pretrained("bert-base-cased"))
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = bert.BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model.to(device)
model.train()

optimizer = AdamW(model.parameters())
for epoch in range(5):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

model.eval()