import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import torch
import os
import json
from nick_stuff.train_loop import EncodingDataset
from nick_stuff.train_loop import OceanEncoder, OceanRegressor

score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']

def parse_pred_ocean(pred_str):
    return [float(val) for val in pred_str.split(',')]

def calc_metrics(test, PATH, EPOCH, RUN_NAME):
    path = PATH + f'/{RUN_NAME}.json'
    score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
    df = test.copy()

    # Create separate columns for each predicted trait
    df[['pred_agreeableness', 'pred_openness', 'pred_conscientiousness', 
        'pred_extraversion', 'pred_neuroticism']] = pd.DataFrame(
        df['pred_ocean'].apply(parse_pred_ocean).tolist(), 
        index=df.index
    )

    # Step 2: Calculate L1 loss (Mean Absolute Error) for each trait
    l1_agreeableness = np.abs(df['agreeableness'] - df['pred_agreeableness']).mean()
    l1_openness = np.abs(df['openness'] - df['pred_openness']).mean()
    l1_conscientiousness = np.abs(df['conscientiousness'] - df['pred_conscientiousness']).mean()
    l1_extraversion = np.abs(df['extraversion'] - df['pred_extraversion']).mean()
    l1_neuroticism = np.abs(df['neuroticism'] - df['pred_neuroticism']).mean()

    # Average L1 loss across all traits
    average_l1_loss = (l1_agreeableness + l1_openness + l1_conscientiousness + 
                      l1_extraversion + l1_neuroticism) / 5

    # Step 3: Calculate L2 loss (Mean Squared Error) for each trait
    l2_agreeableness = np.square(df['agreeableness'] - df['pred_agreeableness']).mean()
    l2_openness = np.square(df['openness'] - df['pred_openness']).mean()
    l2_conscientiousness = np.square(df['conscientiousness'] - df['pred_conscientiousness']).mean()
    l2_extraversion = np.square(df['extraversion'] - df['pred_extraversion']).mean()
    l2_neuroticism = np.square(df['neuroticism'] - df['pred_neuroticism']).mean()

    # Average L2 loss across all traits
    average_l2_loss = (l2_agreeableness + l2_openness + l2_conscientiousness + 
                      l2_extraversion + l2_neuroticism) / 5

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(average_l2_loss)

    # Create metrics dictionary for this epoch
    current_epoch_metrics = {
        'l1_loss': {
            'agreeableness': float(l1_agreeableness),
            'openness': float(l1_openness),
            'conscientiousness': float(l1_conscientiousness),
            'extraversion': float(l1_extraversion),
            'neuroticism': float(l1_neuroticism),
            'average': float(average_l1_loss)
        },
        'l2_loss': {
            'agreeableness': float(l2_agreeableness),
            'openness': float(l2_openness),
            'conscientiousness': float(l2_conscientiousness),
            'extraversion': float(l2_extraversion),
            'neuroticism': float(l2_neuroticism),
            'average': float(average_l2_loss)
        },
        'rmse': float(rmse)
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing data if file exists, or create new structure
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                all_metrics = json.load(f)
            except json.JSONDecodeError:
                # Handle case where file exists but is empty or invalid
                all_metrics = {"epochs": {}}
    else:
        all_metrics = {"epochs": {}}

    # Add or update metrics for this epoch'name'
    all_metrics["epochs"][f"epoch_{EPOCH}"] = current_epoch_metrics

    # Add the latest metrics to a "latest" field for convenience
    all_metrics["latest"] = current_epoch_metrics

    # Save updated metrics to file
    with open(path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print(f"Metrics for epoch {EPOCH} appended to {path}")
    
    return all_metrics
    

    
def calc_responses(questions, name):
    score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
    encoder_model = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    encoder = OceanEncoder(encoder_model=encoder_model)
    model_path = "nick_stuff/saves/ocean_regressor_20250424_131518.pt"
    model = OceanRegressor(encoder, 1024)
    model.load(model_path)
    model.to(device)
    model.eval()

    


    question_df = pd.DataFrame({'text': questions})
    question_df[score_columns] = 0.0

    question_dataset = EncodingDataset(
        dataframe=question_df, 
        tokenizer=tokenizer, 
        encoder=encoder,
        name=name,
        device=device)


    num_batches = len(question_dataset) // 32 + 1
    start_idxs = [32 * i for i in range(num_batches)]
    stop_idxs = [min(32 * (i + 1), len(question_dataset)) for i in range(num_batches)]
    batch_idxs = [list(range(start, stop)) for start, stop in zip(start_idxs, stop_idxs)]
    
    print(len(question_dataset), num_batches)
    score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
    question_df[score_columns] = 0.0
    question_df[['pred_ocean']] = ''
    for k, idx in enumerate(batch_idxs):
        print('batch', k + 1, 'of', len(batch_idxs))

        print('   questions...')
        scores = model(question_dataset.encodings[idx])
        scores = scores.cpu().detach().numpy()
        question_df.loc[idx, score_columns] = scores * 100.0
        question_df.loc[idx, 'pred_ocean'] = [','.join(map(lambda x: f"{x*100:.4f}", row)) for row in scores.tolist()]
    
    question_df = question_df.drop(columns=score_columns)
    return question_df

def compute_ppl(eval_preds):
    print(eval_preds)
    logits, labels = eval_preds
    # shift so that tokens predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}

def compute_metrics(pred):
    predictions = torch.tensor(pred[0])
    targets = torch.tensor(pred[1])
    perplexity = Perplexity(ignore_index=-100)
    perplexity_score = perplexity(predictions, targets)
    return {"Perplexity": perplexity_score}