from train_loop import EncodingDataset
from train_loop import OceanEncoder, OceanRegressor
from transformers import AutoTokenizer
import pandas as pd
import torch


data_file = "gpt_roleplay.csv"
df = pd.read_csv(data_file)

text = []
questions = []
answers = []

for i in df.index:
    for key in ['example_dialogue', 'dialogues']:
        sample = df.iloc[i][key]
        
        substrings = sample.split("\'content\': ")
        substrings = [substring for substring in substrings if "role" in substring]
        substrings = [substring.split(", \'role\':")[0] for substring in substrings]

        substrings_tmp = []
        for substring in substrings:
            if ((substring[0] == "'" or substring[0] == '"') and 
                (substring[-1] == "'" or substring[-1] == '"')):
                substrings_tmp.append(substring[1:-1])
            else:
                substrings_tmp.append(substring)
        substrings = substrings_tmp
        
        if 'content' not in substrings[0]:
            for j in range(len(substrings) - 1):
                if "?" in substrings[j]:
                    questions.append(substrings[j])
                    answers.append(substrings[j + 1])


encoder_model = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(encoder_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")
encoder = OceanEncoder(encoder_model=encoder_model)
model_path = "ocean_regressor_20250424_131518.pt"
model = OceanRegressor()
model.load(model_path)
model.to(device)
model.eval()

question_df = pd.DataFrame({'text': questions})
question_dataset = EncodingDataset(
    dataframe=question_df, 
    tokenizer=tokenizer, 
    encoder=encoder,
    name='question_encodings',
    device=device)

answer_df = pd.DataFrame({'text': answers})
answer_dataset = EncodingDataset(
    dataframe=answer_df, 
    tokenizer=tokenizer, 
    encoder=encoder,
    name='answer_encodings',
    device=device)

num_batches = len(question_dataset) // 32 + 1
start_idxs = [32 * i for i in range(num_batches)]
stop_idxs = [min(32 * (i + 1), len(question_dataset)) for i in range(num_batches)]
batch_idxs = [list(range(start, stop)) for start, stop in zip(start_idxs, stop_idxs)]

score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
question_df[score_columns] = 0.0
answer_df[score_columns] = 0.0

for k, idx in enumerate(batch_idxs):
    print('batch', k + 1, 'of', len(batch_idxs))

    print('   questions...')
    input_ids = question_dataset.input_ids[idx]
    attention_masks = question_dataset.attention_masks[idx]
    scores = model(input_ids, attention_masks)
    scores = scores.cpu().detach().numpy()
    question_df.loc[idx, score_columns] = scores * 100.0

    print('   answers...')
    input_ids = answer_dataset.input_ids[idx]
    attention_masks = answer_dataset.attention_masks[idx]
    scores = model(input_ids, attention_masks)
    scores = scores.cpu().detach().numpy()
    answer_df.loc[idx, score_columns] = scores * 100.0
    
    
    #print(scores)
    #print('')

#for i, idx in enumerate(batch_idxs)
#print('calculating scores...')
#scores = model(input_ids=dataset.input_ids[:32], attention_mask=dataset.attention_masks[:32])
#print(scores)

question_df.to_csv("gpt_roleplay_questions.csv", index=False)
answer_df.to_csv("gpt_roleplay_answers.csv", index=False)
print("Scores saved to gpt_roleplay_scores.csv")
