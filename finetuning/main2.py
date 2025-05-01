# Add automated script to build the model, run it, get scores out, train it, that sort of stuff. 

import pandas as pd
from unsloth.chat_templates import get_chat_template
import math
import numpy as np
import torch
import os
import re
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import wandb
from unsloth.chat_templates import train_on_responses_only
from model import get_model
from error_metrics import calc_metrics, calc_responses, compute_ppl

score_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']

# Will store all of the metrics over here, time, GPU usage also maybe, etc etc
RUN_NUMBER = '2'
RUN_NAME = f"gemma_1b_4bit_v1_{RUN_NUMBER}"
PATH = "saves/" + RUN_NAME
# MODEL_PATH = "saves/" + RUN_NAME + "/" + "gemma_4bit_4b_it_0.pt"
MODEL_PATH = ""
EPOCHS = 1
TRAIN_TEST_SPLIT = 0.99

def apply_chat_template(examples):
    texts = f"<bos><start_of_turn>user\n"
    texts += f"Respond based on the following personality scores:\nagreeableness:{examples['agreeableness']}, openness:{examples['openness']}, conscientiousness:{examples['conscientiousness']}, extraversion:{examples['extraversion']}, neuroticism:{examples['neuroticism']}\n"
    texts += "Here is the question by the user:\n"
    texts += examples["questions"]
    texts += f"<end_of_turn>\n<start_of_turn>model\n"
    texts += examples["answers"]
    texts += "<end_of_turn>\n"

    return texts

def get_output(question_batch, scores_batch):
    messages_batch = [
            [{
            "role": "user",
            "content": [{
                "type" : "text",
                "text" : f"Respond based on the following personality scores:\nagreeableness:{np.round(scores[1][0], 2)}, openness:{np.round(scores[1][1], 2)}, conscientiousness:{np.round(scores[1][2], 2)}, extraversion:{np.round(scores[1][3], 2)}, neuroticism:{np.round(scores[1][4], 2)}\nHere is the question by the user:\n{question}",
            }]
        }] for question, scores in zip(question_batch, scores_batch.iterrows())
    ]
    
    text_to_tokenize = []
    
    for message in messages_batch:
        templated_text = tokenizer.apply_chat_template(
            message,
            add_generation_prompt = True, # Must add for generation
            tokenize=False
        )
        
        text_to_tokenize.append(templated_text)

    text = tokenizer(
        text_to_tokenize,
        return_tensors = "pt",
        padding=True,
        max_length=64
        # add_generation_prompt = True, # Must add for generation
    ).to("cuda")
    
    outputs = model.generate(
        # **tokenizer([text], return_tensors = "pt").to("cuda"),
        **text,
        max_new_tokens = 64, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
    )
    return tokenizer.batch_decode(outputs)



questions = pd.read_csv("nick_stuff/data/gpt_roleplay_questions_v00.csv")
answers = pd.read_csv("nick_stuff/data/gpt_roleplay_answers_v00.csv")
questions = questions.rename(columns={"text": "questions"})
answers = answers.rename(columns={"text": "answers"})

# Lets take the first 90% as train and the rest as test
train = questions.iloc[:int(TRAIN_TEST_SPLIT*len(questions))].copy()
train[answers.columns] = answers.iloc[:int(TRAIN_TEST_SPLIT*len(questions))].copy()

test = questions.iloc[int(TRAIN_TEST_SPLIT*len(questions)):].copy()
test[answers.columns] = answers.iloc[int(TRAIN_TEST_SPLIT*len(questions)):].copy()

test = test.reset_index()

# Add column text using the function apply_chat_template
train['text'] = train.apply(apply_chat_template, axis=1)
dataset = Dataset.from_pandas(train[['text']])

if not os.path.exists(PATH):
    os.makedirs(PATH, exist_ok=True)

# Get the latest model and start training that and then save an updated version of the model
model, tokenizer = get_model(MODEL_PATH)

# Calculate initial
num_batches = len(test) // 128 + 1
start_idxs = [128 * i for i in range(num_batches)]
stop_idxs = [min(128 * (i + 1), len(test)) for i in range(num_batches)]
batch_idxs = [list(range(start, stop)) for start, stop in zip(start_idxs, stop_idxs)]
for r, idx in enumerate(batch_idxs):
    question = test.loc[idx, 'questions']
    scores = test.loc[idx, score_columns]
    
    outputs = get_output(question, scores)
    responses = []
    
    for output in outputs:
        responses.append(output.split("\n<start_of_turn>model\n")[1].replace('<pad>', ''))
    
    test.loc[idx, 'response'] = responses
    
response_df = calc_responses(list(test['response']), RUN_NAME)
test['pred_ocean'] = response_df['pred_ocean']

calc_metrics(test, PATH, "base", RUN_NAME)

wandb.init(
    project="deeplearning-1",
    name=RUN_NAME,
    tags=["SFT", "gemma-3-1b-4bit"],
    config={
        "epochs": EPOCHS,
        "batch_size": 2 * 4,  # effective batch size with gradient accumulation
        "learning_rate": 2e-4,
        "train_test_split": TRAIN_TEST_SPLIT,
        "model": "gemma-3-1b-4bit"
    }
)

test['text'] = test.apply(apply_chat_template, axis=1)
eval_dataset = Dataset.from_pandas(test[['text']])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = eval_dataset,
    compute_metrics=compute_ppl,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = EPOCHS, # Set this for 1 full training run.
        max_steps = 2,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb", # Use this for WandB etc,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)



for i in range(EPOCHS):
    
    
    print("TRAINING EPOCH: ", i)

    trainer.train()
    
    eval_results = trainer.evaluate() 
    print(eval_results)
    loss = eval_results["eval_loss"]                          
    perplexity = math.exp(loss)                           
    print(f">>> Perplexity: {perplexity:.2f}")  
    
    # torch.save(model.state_dict(), f"saves/{RUN_NAME}/{RUN_NAME}.pt")
    model.save_pretrained(f"saves/{RUN_NAME}")
    # Calculate initial
    num_batches = len(test) // 128 + 1
    start_idxs = [128 * i for i in range(num_batches)]
    stop_idxs = [min(128 * (i + 1), len(test)) for i in range(num_batches)]
    batch_idxs = [list(range(start, stop)) for start, stop in zip(start_idxs, stop_idxs)]
    for r, idx in enumerate(batch_idxs):
        question = test.loc[idx, 'questions']
        scores = test.loc[idx, score_columns]

        outputs = get_output(question, scores)
        responses = []

        for output in outputs:
            responses.append(output.split("\n<start_of_turn>model\n")[1].replace('<pad>', ''))

        test.loc[idx, 'response'] = responses

    response_df = calc_responses(list(test['response']), RUN_NAME)
    test['pred_ocean'] = response_df['pred_ocean']

    metrics = calc_metrics(test, PATH, f"{i}", RUN_NAME)
    
    # Log metrics to WandB with epoch as a step
    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            # For nested metrics like l1_loss: {agreeableness: 10, openness: 20, ...}
            for sub_name, sub_value in value.items():
                wandb.log({f"{metric_name}/{sub_name}": sub_value, "epoch": i})
        else:
            wandb.log({metric_name: value, "epoch": i})
            
    # You can also log a summary table of your test results
    if i == EPOCHS - 1:  # On the final epoch
        wandb.log({"final_test_results": wandb.Table(dataframe=test)})



