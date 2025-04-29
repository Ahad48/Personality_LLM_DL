import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm.notebook import tqdm
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

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
    def __init__(self, df, tokenizer, max_length, context_window=None, sos_token=101, qna = False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_lenth = max_length
        self.sos_token = sos_token
        self.qna = qna

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

        tokens = self.tokenizer.encode(text, add_special_tokens = False)
        seq_len = len(tokens)

        
        for i in range(0, seq_len-1):
            # add a sliding window to all the tokens
            if self.context_window+i < seq_len - 1:
                encoder_input = tokens[i:self.context_window+i]
                # shiftign the output by 1 and using is as expected output
                expected_output = tokens[i+1:self.context_window+i+1]
            
            else:
                
                encoder_input = tokens[i:-1]
                expected_output = tokens[i+1:]

            # encoder_input = self.tokenizer.encode(encoder_input, add_special_tokens = True)
            # print(encoder_input)
            # print(expected_output)
            # print()
            self.processed_data.append({
                    'encoder_input': [self.sos_token] + encoder_input + [self.tokenizer.sep_token_id],
                    'expected_output': [self.sos_token] + expected_output + [self.tokenizer.sep_token_id],
                    'personality': personality
                })
    
    def process_qna(self, question, answer, personality):
        
        encoder_input = self.tokenizer.encode(question, add_special_tokens = False)
        expected_output = self.tokenizer.encode(answer, add_special_tokens = False)
        if not isinstance(personality, torch.Tensor):
            personality = torch.tensor(personality)


        self.processed_data.append(
            {
                'encoder_input': [self.sos_token] + encoder_input + [self.tokenizer.sep_token_id],
                'expected_output': [self.sos_token] + expected_output + [self.tokenizer.sep_token_id],
                'personality': personality
            }
        )

        

    def process_df(self):
        if self.qna:
            self.df.apply(lambda x: self.process_qna(x['question'], x['answer'], x['personality']), axis = 1)
        else:
            self.df.apply(lambda x: self.process_text(x['text'], x['personality']), axis = 1)

def align_batch_data(batch_seq, max_length, pad_token, device):
    def pad_and_get_item(key_val, convert = True):
        if convert:
            seq = [torch.tensor(x[key_val]) for x in batch_seq]
        else:
            seq = [x[key_val] for x in batch_seq]
        padded_seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=pad_token)

        return padded_seq
    

    encoder_inputs = pad_and_get_item('encoder_input')
    expected_outputs = pad_and_get_item('expected_output')
    personalities = pad_and_get_item('personality', convert=False) # no need to convert personality to tensor as already a tensor
    final_dict = {
        'encoder_input':encoder_inputs.to(device),
        'expected_output':expected_outputs.to(device),
        'personality':personalities.to(device)
    }
    return final_dict

def create_batch_data(df, tokenizer, batch_size, max_length, qna = False, context_window = None, pad_token = 0, sos_token=101, device = 'mps'):
    if qna:
        data_obj = QNADataset(df, tokenizer, max_length, device=device)
        data_loader = DataLoader(data_obj, batch_size, shuffle=True)
    else:
        data_obj = PersonalityTextDataset(df, tokenizer, max_length, context_window, sos_token)
        data_loader = DataLoader(data_obj, batch_size, True, collate_fn = lambda x: align_batch_data(x, max_length, pad_token, device = device))

    return data_loader

def train(model, data_loader, optimizer, criterion , device):
    model.train()
    total_loss = 0
    for data in (t_bar:= tqdm(data_loader, desc = "Train Set", leave=False)):
        
        out = model(data["encoder_input"], data['expected_output'], data['personality'])
        out = out.reshape(-1, out.shape[-1])
        
        optimizer.zero_grad()
        loss = criterion(out, data['expected_output'].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss+=loss.item()

        t_bar.set_postfix_str(f'Current Perplexity: {torch.exp(loss)}')

    return total_loss, total_loss/len(data_loader)

def eval_model(model, data_loader, criterion, device):
    
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data in tqdm(data_loader,desc = "Eval Set", leave=False):
            out = model(data["encoder_input"], data['expected_output'], data['personality'])
            out = out.reshape(-1, out.shape[-1])
            loss = criterion(out, data['expected_output'].reshape(-1))

            total_loss += loss.item()

    return total_loss, total_loss/len(data_loader)

class QNADataset(Dataset):
    def __init__(self, df, tokenizer, max_length = 512,question_max_length=None, answer_max_length=None, device = 'cpu'):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.q_max_length = question_max_length
        self.a_max_length = answer_max_length
        self.device = device
        # self.process_text()

    def __len__(self):
        return self.df.shape[0]

    # Using a similar approach for using pandas directly with pytorch dataset https://stackoverflow.com/a/74594835
    def __getitem__(self, index):
        question = self.df['question'].iloc[index]
        answer = self.df['answer'].iloc[index]
        personality = self.df['personality'].iloc[index]

        input_text = f"Question: {question} \nAnswer: {answer}"
        out = self.tokenizer(input_text, max_length=self.max_length, padding = "max_length", add_special_tokens=True, return_tensors = "pt")
        input_tokens, input_mask = out['input_ids'], out['attention_mask']

        out = {
            'input_token':input_tokens.squeeze(0),
            'input_mask': input_mask.squeeze(0),
            'personality': personality
        }
        return out



    def process_text(self):

        def get_tokenized_text(text, tokenizer, max_length, question = True):
            if question:
                text = f"Question: {text} \nAnswer:"

            out = tokenizer(text, max_length=max_length, padding = "max_length", add_special_tokens=True)
            return out['input_ids'], out['attention_mask']

        self.df[['question_token', 'question_mask']] = self.df.apply(lambda x: get_tokenized_text(x['question'],
                                                    self.tokenizer, max_length=self.q_max_length), axis = 1, result_type="expand")

        self.df[['answer_token', 'answer_mask']] = self.df.apply(lambda x: get_tokenized_text(x['answer'],
                                                    self.tokenizer, max_length=self.a_max_length, questions = False), axis = 1, result_type="expand")

        

        
    

    
