import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


def find_backend():
    # if you want to default to cuda first change order.
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Currently using: ", device)
    return device

def tokenize_column(column, tokenizer, max_length=None):
    if max_length!=None:    
        return column.apply(lambda x:tokenizer.encode(x[:max_length], max_length = max_length))
    else:
        return column.apply(lambda x:tokenizer.tokenize(x))
    
def encode_text(df, tokenizer, max_length):
    df["tokenized"] = tokenize_column(df["text"], tokenizer)
    df["encoded"] = tokenize_column(df["tokenized"], tokenizer, max_length=max_length)
    return df

class PersonalityTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, context_window=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_lenth = max_length

        # take full sentence if the window is not specified
        self.context_window = context_window if context_window != None else max_length
        self.processed_data = []
        self.process_text()

    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index):
        return self.processed_data[index]
    
    def process_text(self, text, personality):

        if not isinstance(personality, torch.Tensor):
            personality = torch.tensor(personality)

        tokens = self.tokenizer.encode(text, add_special_tokens = True,
                                       max_length = self.max_lenth,
                                    )
        seq_len = tokens.shape[0]

        if 1 < seq_len < self.context_window + 1:
            


    
        

        
    

    
