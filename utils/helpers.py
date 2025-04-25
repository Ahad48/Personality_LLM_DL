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
    def __init__(self, df, tokenizer, max_length, context_window=None, sos_token=101):
        self.df = df
        self.tokenizer = tokenizer
        self.max_lenth = max_length
        self.sos_token = sos_token

        # take full sentence if the window is not specified
        self.context_window = context_window if context_window != None else max_length
        self.context_window -= 2 # to allow for special tokens sos and eos
        self.processed_data = []
        self.process_df()

    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index):
        return self.processed_data[index]
    
    def process_text(self, text, personality):

        if not isinstance(personality, torch.Tensor):
            personality = torch.tensor(personality)

        tokens = self.tokenizer.tokenize(text)
        seq_len = tokens.shape[0]

        if 1 < seq_len < self.context_window + 1:
            # select all tokens except the last
            encoder_input = tokens[:-1]
            expected_output = tokens[-1:]

        else:
            for i in range(0, seq_len - self.context_window):
                # add a sliding window to all the tokens
                encoder_input = tokens[i:self.context_window+i]
                # shiftign the output by 1 and using is as expected output
                expected_output = tokens[i+1:self.context_window+1]
                
        # decoder input would be random tokens
        decoder_input = torch.randint(0,self.tokenizer.vocab_size - 1 , encoder_input.shape)
        decoder_input[0] = self.sos_token

        self.processed_data.append({
                'encoder_input': self.tokenizer.encode(encoder_input, add_special_tokens = True),
                'decoder_input': decoder_input,
                'expected_output': self.tokenizer.encode(expected_output, add_special_tokens = True),
                'personality': personality
            })
        
    def process_df(self):
        self.df.apply(lambda x: self.process_text(x.text, x.personality), axis = 1)

def align_batch_data(batch_seq, max_length, pad_token):
    def pad_and_get_item(key_val):
        seq = [x['key_val'] for x in batch_seq]
        seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=pad_token)

    encoder_inputs = pad_and_get_item('encoder_input')
    decoder_inputs = pad_and_get_item('decoder_input')
    expected_outputs = pad_and_get_item('expected_output')
    personalities = pad_and_get_item('personality')

    final_dict = {
        'encoder_input':encoder_inputs, 
        'decoder_input':decoder_inputs,
        'expected_output':expected_outputs,
        'personality':personalities
    }
    return final_dict
    

    


    
        

        
    

    
