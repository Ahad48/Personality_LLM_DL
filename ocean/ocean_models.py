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


class OceanEncoder(nn.Module):

    def __init__(self, encoder_model):
        super().__init__()
        self.model = RobertaModel.from_pretrained(encoder_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)        
        pooled = outputs.last_hidden_state[:, 0]
        return pooled
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class OceanRegressor(nn.Module):

    def __init__(self, encoder, size, dropout=0.1):
        super().__init__()
        hidden_size1 = encoder.model.config.hidden_size
        hidden_size2 = size

        self.regressor = nn.Sequential(

            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size2),

            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size2),

            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size2),

            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size2),

            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size2),

            nn.Linear(hidden_size2, 5),
            nn.Sigmoid())

    def forward(self, x):

        x = self.regressor[0](x)  # First layer
        skip_connection = x  # Store the input for the skip connection

        for layer in self.regressor[1:-1]:
            x = layer(x)
            if isinstance(layer, nn.LayerNorm):  
                # Apply skip connection after LayerNorm
                x = x + skip_connection
                skip_connection = x  
                # Update skip connection for the next block

        out = self.regressor[-1](x)  # Last layer
        return out
    
    def save(self, timestamp):
        path = f"saves/ocean_regressor_{timestamp}.pt"
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class OceanModel(nn.Module):

    def __init__(self, encoder, regressor):
        super().__init__()
        self.encoder = encoder
        self.regressor = regressor

    def forward(self, input_ids, attention_mask):
        encodings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.regressor(encodings)
        return outputs
    
    def save(self, timestamp):
        path = f"saves/ocean_model_{timestamp}.pt"
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()