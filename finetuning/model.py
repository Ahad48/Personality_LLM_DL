
from unsloth import FastModel
import torch
from peft import PeftModel

def get_model(MODEL_PATH):
    model, tokenizer = FastModel.from_pretrained(
            model_name = "unsloth/gemma-3-4b-it",
            max_seq_length = 2048, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )
    if len(MODEL_PATH) == 0:
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers     = False, # Turn off for just text!
            finetune_language_layers   = True,  # Should leave on!
            finetune_attention_modules = True,  # Attention good for GRPO
            finetune_mlp_modules       = True,  # SHould leave on always!

            r = 8,           # Larger = higher accuracy, but might overfit
            lora_alpha = 8,  # Recommended alpha == r at least
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
        )
    else:
        model = PeftModel.from_pretrained(model, MODEL_PATH)
    
    
    return model, tokenizer
