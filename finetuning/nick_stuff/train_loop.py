from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import time as time

from nick_stuff.data import retrieve_data, TextDataset, EncodingDataset
from nick_stuff.models import OceanEncoder, OceanRegressor, OceanModel
from nick_stuff.utils import train


encoder_model = "distilroberta-base"
from_text = False
max_length = 514 

epochs = 250
size = 1024  # 2048
batch_size = 64
learning_rate = 1e-4 # 5e-4 # 5e-4
dropout = 0.0
weight_decay = 0.0

load = False # ""


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}\n")

    print("initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    print("   complete.\n")

    print("initializing encoder...")
    encoder = OceanEncoder(encoder_model=encoder_model)
    print("   complete.\n")

    print("initializing regressor...")
    regressor = OceanRegressor(encoder=encoder, size=size, dropout=dropout)
    print("   complete.\n")

    print("downloading datasets...")
    train_df, valid_df, test_df = retrieve_data()
    print("   complete.\n")

    if from_text:

        print('tokeninzing input text...')
        train_dataset = TextDataset(train_df, tokenizer=tokenizer, device=device)
        valid_dataset = TextDataset(valid_df, tokenizer=tokenizer, device=device)
        test_dataset = TextDataset(test_df, tokenizer=tokenizer, device=device)
        print('   complete.\n')

        model = OceanModel(encoder=encoder, regressor=regressor)

    else:

        print('encoding input text...')
        train_dataset = EncodingDataset(train_df, 
            tokenizer=tokenizer, 
            encoder=encoder, 
            name='train_encodings', 
            device=device)
        valid_dataset = EncodingDataset(valid_df, 
            tokenizer=tokenizer, 
            encoder=encoder, 
            name='valid_encodings',
            device=device)
        test_dataset = EncodingDataset(test_df, 
            tokenizer=tokenizer, 
            encoder=encoder, 
            name='test_encodings',
            device=device)
        print('   complete.\n')

        model = regressor

    print('loading model...')
    if load != False:
        print(f'   loading model from {load}')
        path = f"saves/{load}"
        model.load(path)
    model = model.to(device)
    print('complete.\n')

    print('preparing data loaders...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('   complete.\n')

    print('setting up training...')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.MSELoss()
    print('   complete.\n')

    train(
        epochs=epochs, 
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler, 
        criterion=criterion,
        from_text=from_text)

#model_path = r"C:\Users\nickc\Documents\ocean\ocean_regressor_20250415_205250.pt"
#model.load(model_path)