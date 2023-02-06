import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.models import bert
from datasets import DatasetDict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CustomDataset(Dataset):
    def __init__(self, fp):
        self.fp = fp
        self.data = pd.read_csv(self.fp)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        self.x = self.data.iloc[1:701, 0]
        self.y = self.data.iloc[1:701, 1]
        
        # Want integer values instead of 'positive'/'negative'
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        inputs = self.tokenizer(self.x[index], truncation=True, padding='max_length')
        # inputs = torch.tensor(inputs.encodings)
        # I can't convert input into tensor
        
        # labels = self.tokenizer(self.y[index], truncation=True, padding=True)
        labels = 1 if self.y[index] == 'positive' else 0
        inputs['labels'] = torch.tensor(labels)
        return inputs

dataset = CustomDataset('IMDB_Dataset.csv')

train_set, eval_set = random_split(dataset, [600, 100])

training_args = TrainingArguments(output_dir='./results',
                                  num_train_epochs=3,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=10,
                                  warmup_steps=500,
                                  weight_decay=0.01,
                                  logging_dir='./logs',
                                  logging_steps=10)

model = bert.BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_set,
                  eval_dataset = eval_set)

trainer.train()
    