import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import RobertaModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook

import tqdm
import time as time
from datetime import datetime
import random
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import os


def get_batches(dataset, batch_size=32):
    num_batches = len(dataset) // batch_size + 1
    start_idxs = [batch_size * i for i in range(num_batches)]
    stop_idxs = [min(batch_size * (i + 1), len(dataset)) for i in range(num_batches)]
    batch_idxs = [list(range(start, stop)) for start, stop in zip(start_idxs, stop_idxs)]
    return batch_idxs


def retrieve_data():
    splits = {
        'train': 'Personality Datasets - Reddit/train_set.csv', 
        'validation': 'Personality Datasets - Reddit/val_set.csv', 
        'test': 'Personality Datasets - Reddit/eval_set.csv'}
    train_df = pd.read_csv("hf://datasets/Fatima0923/Automated-Personality-Prediction/" + splits["train"])
    valid_df = pd.read_csv("hf://datasets/Fatima0923/Automated-Personality-Prediction/" + splits["validation"])
    test_df  = pd.read_csv("hf://datasets/Fatima0923/Automated-Personality-Prediction/" + splits["test"])
    return train_df, valid_df, test_df


class TextDataset(Dataset):

    def __init__(self, dataframe, tokenizer, device, add_scores=True):

        self.length = len(dataframe)

        self.input_ids = []
        self.attention_masks = []
        for text in dataframe['text']:
            tokenizer_out = tokenizer(text, max_length=tokenizer.model_max_length, padding='max_length', truncation=True)
            self.input_ids.append(tokenizer_out['input_ids'])
            self.attention_masks.append(tokenizer_out['attention_mask'])

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.input_ids = self.input_ids.to(device)

        self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.long)
        self.attention_masks = self.attention_masks.to(device)

        if add_scores:
            self.has_scores = True
            score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
            self.scores = torch.tensor(dataframe[score_columns].values, dtype=torch.float) / 100.0
            self.scores = self.scores.to(device)
        else:
            self.has_scores = False
            self.scores = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        out = {
            'input_ids': self.input_ids[idx], 
            'attention_mask': self.attention_masks[idx]}
        if self.has_scores:
            out['scores'] = self.scores[idx]
        return out
    

class EncodingDataset(Dataset):

    def __init__(self, dataframe, tokenizer, encoder, device, name, add_scores=True):
        path = f"saves/{name}/{name}.pt"
        # if os.path.exists(path):
        #     print(f"Loading precomputed encodings from {path}")
        #     data = torch.load(path)
        #     self.encodings = data['encodings']
        #     self.length = self.encodings.shape[0]
        #     self.width = self.encodings.shape[1]

        if True:

            encoder = encoder.to(device)
            encoder.eval()

            max_length = tokenizer.model_max_length
            self.length = len(dataframe)
            self.width = encoder.model.config.hidden_size
            self.encodings = torch.zeros((self.length, self.width), dtype=torch.float, device='cpu')

            batch_idxs = get_batches(dataframe, batch_size=64)
            for k, idx in enumerate(batch_idxs):
                #if k % 10 == 0:
                print(f'batch {k + 1} of {len(batch_idxs)}')
                print("   torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                
                input_ids = []
                attention_masks = []
                for k, text in enumerate(dataframe['text'][idx]):                
                    tokenizer_out = tokenizer(text, max_length=max_length, padding='max_length', truncation=True)
                    input_ids.append(tokenizer_out['input_ids'])
                    attention_masks.append(tokenizer_out['attention_mask'])
                
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
                attention_masks = torch.tensor(attention_masks, dtype=torch.long).to(device)
                with torch.no_grad():
                    encoder_out = encoder(input_ids=input_ids, attention_mask=attention_masks) 
                encoder_out = encoder_out.to('cpu')
                self.encodings[idx] = encoder_out

            torch.save({'encodings': self.encodings}, path)

        self.encodings = self.encodings.to(device)

        if add_scores:
            self.has_scores = True
            score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
            self.scores = torch.tensor(dataframe[score_columns].values, dtype=torch.float) / 100.0
            self.scores = self.scores.to(device)
        else:
            self.has_scores = False
            self.scores = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        out = {'encodings': self.encodings[idx]}
        if self.has_scores:
            out['scores'] = self.scores[idx]
        return out