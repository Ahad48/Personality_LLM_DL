
def get_chat_template_for_model(model_name):
    """
    Returns the appropriate chat template function based on model name
    
    Args:
        model_name: Name of the model
    
    Returns:
        apply_chat_template function specific to the model
    """
    
    if "gemma" in model_name.lower():
        def apply_chat_template(examples):
            texts = f"<bos><start_of_turn>user\n"
            texts += f"Respond based on the following personality scores:\nagreeableness:{examples['agreeableness']}, openness:{examples['openness']}, conscientiousness:{examples['conscientiousness']}, extraversion:{examples['extraversion']}, neuroticism:{examples['neuroticism']}\n"
            texts += "Here is the question by the user:\n"
            texts += examples["questions"]
            texts += f"<end_of_turn>\n<start_of_turn>model\n"
            texts += examples["answers"]
            texts += "<end_of_turn>\n"
            return texts
            
        def get_response_prefix(tokenizer):
            return "\n<start_of_turn>model\n"
            
        def get_instruction_prefix(tokenizer):
            return "<start_of_turn>user\n"
            
    elif "smollm" in model_name.lower() or "small-m" in model_name.lower():
        def apply_chat_template(examples):
            texts = f"<|im_start|>user\n"
            texts += f"Respond based on the following personality scores:\nagreeableness:{examples['agreeableness']}, openness:{examples['openness']}, conscientiousness:{examples['conscientiousness']}, extraversion:{examples['extraversion']}, neuroticism:{examples['neuroticism']}\n"
            texts += "Here is the question by the user:\n"
            texts += examples["questions"]
            texts += f"<|im_end|>\n<|im_start|>assistant\n"
            texts += examples["answers"]
            texts += "<|im_end|>\n"
            return texts
            
        def get_response_prefix(tokenizer):
            return "\n<|im_start|>assistant\n"
            
        def get_instruction_prefix(tokenizer):
            return "<|im_start|>user\n"
            
    elif "mistral" in model_name.lower():
        def apply_chat_template(examples):
            texts = f"<s>[INST] "
            texts += f"Respond based on the following personality scores:\nagreeableness:{examples['agreeableness']}, openness:{examples['openness']}, conscientiousness:{examples['conscientiousness']}, extraversion:{examples['extraversion']}, neuroticism:{examples['neuroticism']}\n"
            texts += "Here is the question by the user:\n"
            texts += examples["questions"]
            texts += f" [/INST] "
            texts += examples["answers"]
            texts += "</s>"
            return texts
            
        def get_response_prefix(tokenizer):
            return " [/INST] "
            
        def get_instruction_prefix(tokenizer):
            return "[INST] "
            
    elif "llama" in model_name.lower():
        def apply_chat_template(examples):
            texts = f"<s>[INST] "
            texts += f"Respond based on the following personality scores:\nagreeableness:{examples['agreeableness']}, openness:{examples['openness']}, conscientiousness:{examples['conscientiousness']}, extraversion:{examples['extraversion']}, neuroticism:{examples['neuroticism']}\n"
            texts += "Here is the question by the user:\n"
            texts += examples["questions"]
            texts += f" [/INST] "
            texts += examples["answers"]
            texts += "</s>"
            return texts
            
        def get_response_prefix(tokenizer):
            return " [/INST] "
            
        def get_instruction_prefix(tokenizer):
            return "[INST] "
    
#     else:
#         # Default to using the model's built-in chat template
#         def apply_chat_template(examples):
#             messages = [
#                 {"role": "user", "content": f"Respond based on the following personality scores:\nagreeableness:{examples['agreeableness']}, openness:{examples['openness']}, conscientiousness:{examples['conscientiousness']}, extraversion:{examples['extraversion']}, neuroticism:{examples['neuroticism']}\nHere is the question by the user:\n{examples['questions']}"},
#                 {"role": "assistant", "content": examples["answers"]}
#             ]
#             return tokenizer.apply_chat_template(messages, tokenize=False)
            
#         def get_response_prefix(tokenizer):
#             # This is a placeholder - the actual prefix will be determined by the tokenizer's chat template
#             return None
            
#         def get_instruction_prefix(tokenizer):
#             # This is a placeholder - the actual prefix will be determined by the tokenizer's chat template
#             return None
            
    return apply_chat_template, get_response_prefix, get_instruction_prefix


def format_input_for_model(question, scores, model_name, tokenizer):
    """
    Format input based on the model's chat template
    """
    if "gemma" in model_name.lower():
        message = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Respond based on the following personality scores:\nagreeableness:{np.round(scores[1][0], 2)}, openness:{np.round(scores[1][1], 2)}, conscientiousness:{np.round(scores[1][2], 2)}, extraversion:{np.round(scores[1][3], 2)}, neuroticism:{np.round(scores[1][4], 2)}\nHere is the question by the user:\n{question}",
            }]
        }]
        
    elif "smallm" in model_name.lower() or "small-m" in model_name.lower():
        message = [{
            "role": "user",
            "content": f"Respond based on the following personality scores:\nagreeableness:{np.round(scores[1][0], 2)}, openness:{np.round(scores[1][1], 2)}, conscientiousness:{np.round(scores[1][2], 2)}, extraversion:{np.round(scores[1][3], 2)}, neuroticism:{np.round(scores[1][4], 2)}\nHere is the question by the user:\n{question}"
        }]
        
    elif "mistral" in model_name.lower() or "llama" in model_name.lower():
        message = [{
            "role": "user",
            "content": f"Respond based on the following personality scores:\nagreeableness:{np.round(scores[1][0], 2)}, openness:{np.round(scores[1][1], 2)}, conscientiousness:{np.round(scores[1][2], 2)}, extraversion:{np.round(scores[1][3], 2)}, neuroticism:{np.round(scores[1][4], 2)}\nHere is the question by the user:\n{question}"
        }]
        
    else:
        message = [{
            "role": "user",
            "content": f"Respond based on the following personality scores:\nagreeableness:{np.round(scores[1][0], 2)}, openness:{np.round(scores[1][1], 2)}, conscientiousness:{np.round(scores[1][2], 2)}, extraversion:{np.round(scores[1][3], 2)}, neuroticism:{np.round(scores[1][4], 2)}\nHere is the question by the user:\n{question}"
        }]
        
    # Apply the model's tokenizer chat template
    return tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)


def extract_response(output_text, model_name):
    """
    Extract the model's response from the output text based on the model type
    """
    if "gemma" in model_name.lower():
        return output_text.split("\n<start_of_turn>model\n")[1].replace('<pad>', '')
    elif "smallm" in model_name.lower() or "small-m" in model_name.lower():
        return output_text.split("\n<|im_start|>assistant\n")[1].split("<|im_end|>")[0].replace('<pad>', '')
    elif "mistral" in model_name.lower() or "llama" in model_name.lower():
        return output_text.split(" [/INST] ")[1].replace('</s>', '').replace('<pad>', '')
    else:
        # Default case - try to use a common pattern or let the caller handle
        try:
            return output_text.split("assistant")[1].replace('<pad>', '')
        except:
            return output_text.replace('<pad>', '')