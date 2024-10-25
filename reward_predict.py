import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Qwen2Tokenizer
from template import template_dict
import json

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def reward_value(question, answer, model, tokenizer, max_length=4096, template_name='qwen'):
    # chat template
    template = template_dict[template_name]
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = template.system
    if system:
        system_text = system_format.format(content=system)
    user_text = user_format.format(content=question)
    prompt = system_text + user_text 

    # predict
    prompt = prompt + assistant_format.format(content=answer, stop_token=tokenizer.eos_token)
    inputs = tokenizer(prompt, max_length=max_length, truncation=True, padding=True, return_tensors='pt').to(model.device)
    output = model(**inputs)
    logits = output['logits']
    score = torch.sigmoid(logits).tolist()[0][0]
    return score
