import os
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from datetime import datetime

def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["text_target"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- parameters ----
dataset_path = "data"
questions_file = "data/gpt_roleplay_questions_v00.csv"
answers_file = "data/gpt_roleplay_answers_v00.csv"
transformer_model = "t5-small"
test_size = 0.10
batch_size = 32
epochs = 5

tokenizer = AutoTokenizer.from_pretrained(transformer_model)
questions_df = pd.read_csv(questions_file)
answers_df = pd.read_csv(answers_file)
questions_df["text_target"] = answers_df["text"]
dataset = Dataset.from_pandas(questions_df[["text", "text_target"]])
tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=test_size)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

model = AutoModelForSeq2SeqLM.from_pretrained(transformer_model)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"saves/finetuned_t5_small_model_{timestamp}"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    #evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator)

trainer.train()

