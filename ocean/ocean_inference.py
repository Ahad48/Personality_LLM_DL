from ocean_trainer import EncodingDataset
from ocean_trainer import OceanEncoder, OceanRegressor
from transformers import AutoTokenizer
import pandas as pd
import torch


# --- parameters ---
data_file = "datasets/gpt_roleplay_raw.csv"
encoder_model = "distilroberta-base"
model_path = "saves/ocean_regressor_20250425_123222.pt"
version = "v01"

print("loading data...")
df = pd.read_csv(data_file)
print("   complete.\n")

print("processing data...")
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

print("   found", len(questions), "questions and", len(answers), "answers.\n")
print("   complete.\n")

print("loading tokenizer and encoder...")
tokenizer = AutoTokenizer.from_pretrained(encoder_model)
encoder = OceanEncoder(encoder_model=encoder_model)
print("   complete.\n")

print("loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   using device: {device}")
model = OceanRegressor(encoder=encoder, size=1024, dropout=0.0)
model.load(model_path)
model.to(device)
model.eval()
print("   complete.\n")

print('encoding questions...')
question_df = pd.DataFrame({'text': questions})
question_dataset = EncodingDataset(
    dataframe=question_df, 
    tokenizer=tokenizer, 
    encoder=encoder,
    name='gpt_roleplay_question_encodings',
    device=device,
    add_scores=False)
print('   complete.\n')

print('encoding answers...')
answer_df = pd.DataFrame({'text': answers})
answer_dataset = EncodingDataset(
    dataframe=answer_df, 
    tokenizer=tokenizer, 
    encoder=encoder,
    name='gpt_roleplay_answer_encodings',
    device=device,
    add_scores=False)
print('   complete.\n')


num_batches = len(question_dataset) // 32 + 1
start_idxs = [32 * i for i in range(num_batches)]
stop_idxs = [min(32 * (i + 1), len(question_dataset)) for i in range(num_batches)]
batch_idxs = [list(range(start, stop)) for start, stop in zip(start_idxs, stop_idxs)]

score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
question_df[score_columns] = 0.0
answer_df[score_columns] = 0.0

print("scoring questions and answers...")
for k, idx in enumerate(batch_idxs):
    print('batch', k + 1, 'of', len(batch_idxs))

    print('   questions...')
    #input_ids = question_dataset.input_ids[idx]
    #attention_masks = question_dataset.attention_masks[idx]
    #scores = model(input_ids, attention_masks)
    encodings = question_dataset.encodings[idx]
    scores = model(encodings)
    scores = scores.cpu().detach().numpy()
    question_df.loc[idx, score_columns] = scores * 100.0

    print('   answers...')
    #input_ids = answer_dataset.input_ids[idx]
    #attention_masks = answer_dataset.attention_masks[idx]
    #scores = model(input_ids, attention_masks)
    encodings = answer_dataset.encodings[idx]
    scores = model(encodings)
    scores = scores.cpu().detach().numpy()
    answer_df.loc[idx, score_columns] = scores * 100.0

question_df.to_csv("datasets/gpt_roleplay_questions_" + version + ".csv", index=False)
answer_df.to_csv("datasets/gpt_roleplay_answers_" + version + ".csv", index=False)
print("scores saved to datasets folder.\n")