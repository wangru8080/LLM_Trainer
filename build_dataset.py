import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
import logging
from datasets import load_dataset, concatenate_datasets
import transformers
import itertools
import functools
from tqdm import tqdm
from template import template_dict

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

def messages_tokenization(examples, tokenizer, max_seq_len, template_name, discard_long_sample):
    if template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[template_name]
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = template.system

    len_sorted = []
    for messages in examples['conversations']:
        system_flag = 0
        input_ids, labels = [], []
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                system_flag = 1
                system_text = system_format.format(content=content)
                system_tokens = tokenizer.encode(system_text, add_special_tokens=False)
                input_ids += system_tokens
                labels += [IGNORE_INDEX] * len(system_tokens)
            elif role == 'user':
                user_text = user_format.format(content=content, stop_token=tokenizer.eos_token)
                input_tokens = tokenizer.encode(user_text, add_special_tokens=False)
                input_ids += input_tokens
                labels += [IGNORE_INDEX] * len(input_tokens)
            elif role == 'assistant':
                assistant_text = assistant_format.format(content=content, stop_token=tokenizer.eos_token)
                output_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
                input_ids += output_tokens
                labels += output_tokens

        if system_flag == 0: # 数据中不包含system
            if system:
                system_text = system_format.format(content=system)
                system_tokens = tokenizer.encode(system_text, add_special_tokens=False)
                input_ids = system_tokens + input_ids
                labels = [IGNORE_INDEX] * len(system_tokens) + labels
        
        if discard_long_sample and len(labels) > max_seq_len:
            continue
        
        len_sorted.append([input_ids, labels])

        len_sorted.sort(key=lambda x: len(x[0]))

    all_input_ids = []
    all_labels = []
        
    for ids, labels in len_sorted:
        all_input_ids.append(ids[:max_seq_len])
        all_labels.append(labels[:max_seq_len])
    
    return {'input_ids': all_input_ids, 'labels': all_labels}

def build_sft_dataset(data_path: Union[List[str], str],
                      tokenizer: transformers.PreTrainedTokenizer,
                      max_seq_len: int, 
                      discard_long_sample: bool,
                      data_cache_dir=None,
                      preprocessing_num_workers=None,
                      template_name='qwen'
                      ):
    logging.info('building dataset...')

    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]

    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))

        cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets - {file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            raw_dataset = raw_dataset.select_columns(['conversations'])

            tokenization = functools.partial(messages_tokenization, tokenizer=tokenizer, max_seq_len=max_seq_len, template_name=template_name, discard_long_sample=discard_long_sample)

            processed_dataset = raw_dataset.map(
                tokenization,
                batched=True,
                batch_size=8000,
                num_proc=preprocessing_num_workers,
                remove_columns=['conversations'],
                keep_in_memory=False,
                desc='preprocessing on dataset',
            )
            processed_dataset.save_to_disk(cache_path)

        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

def tokenize_function(examples, tokenizer):
    input_ids = []
    for example in examples['text']:
        output = tokenizer.encode(example, add_special_tokens=False)
        output.append(tokenizer.eos_token_id)
        input_ids.append(output)
    labels = input_ids.copy()
    return {'input_ids': input_ids, 'labels': labels}

def group_texts(examples, max_seq_len):
    # Concatenate all texts.
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
    total_len = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_len >= max_seq_len:
        total_len = (total_len // max_seq_len) * max_seq_len
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_seq_len] for i in range(0, total_len, max_seq_len)]
        for k, t in concatenated_examples.items()
    }
    return result

def build_pretrain_dataset(data_path: Union[List[str], str],
                           tokenizer: transformers.PreTrainedTokenizer,
                           max_seq_len: int, data_cache_dir = None,
                           preprocessing_num_workers = None
                           ):
    logging.info('building pretrain dataset...')

    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]

    for file in tqdm(data_path):
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))

        cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)

        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets - {file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset('json', data_files=file, cache_dir=cache_path)
            raw_dataset = raw_dataset.select_columns(['text'])
            tokenize_func = functools.partial(tokenize_function, tokenizer=tokenizer)
            processed_dataset = raw_dataset.map(
                tokenize_func,
                batched=True,
                batch_size=8000,
                num_proc=preprocessing_num_workers,
                remove_columns=['text'],
                keep_in_memory=False,
                desc='Running tokenizer on dataset',
            )
            group_texts_func = functools.partial(group_texts, max_seq_len=max_seq_len)
            processed_dataset = processed_dataset.map(
                group_texts_func,
                batched=True,
                batch_size=8000,
                num_proc=preprocessing_num_workers,
                keep_in_memory=False,
                desc='Grouping texts in chunks of {max_seq_len}',
            )
            processed_dataset.save_to_disk(cache_path)

        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    logger.info(f'Total training number: {len(all_datasets)}')
    return all_datasets

def return_prompt_and_responses(examples, template_name, max_seq_len: int):
    if template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[template_name]
    system_format = template.system_format
    user_format = template.user_format
    system = template.system
    
    prompt_list = []
    chosen_list = []
    rejected_list = []
    for question, response_chosen, response_rejected in zip(examples['question'], examples['response_chosen'], examples['response_rejected']):
        system_text = ''
        if system:
            system_text = system_format.format(content=system)
        user_text = user_format.format(content=question)
        question = system_text + user_text

        if len(question) + len(response_chosen) > max_seq_len or len(question) + len(response_rejected) > max_seq_len:
            continue
        if len(question) == 0 or len(response_chosen) == 0 or len(response_rejected) == 0:
            continue
        prompt_list.append(question)
        chosen_list.append(response_chosen)
        rejected_list.append(response_rejected)

    return {
        'prompt': prompt_list,
        'chosen': chosen_list,
        'rejected': rejected_list,
    }

def build_dpo_dataset(data_path: Union[List[str], str], max_seq_len: int, data_cache_dir=None, preprocessing_num_workers=None, template_name='qwen'):
    logger.info('building dataset...')

    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]
    
    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets - {file} has been loaded from disk')
        except:
            raw_dataset = load_dataset('json', data_files=file, cache_dir=cache_path)
            raw_dataset = raw_dataset.select_columns(['question', 'response_chosen', 'response_rejected'])
            map_function = functools.partial(return_prompt_and_responses, template_name=template_name, max_seq_len=max_seq_len)
            processed_dataset = raw_dataset.map(
                map_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=['question', 'response_chosen', 'response_rejected'],
                keep_in_memory=False,
                desc='preprocessing on dataset'
            )
            processed_dataset.save_to_disk(cache_path)
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForPadding(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ('input_ids', 'labels'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import glob
    tokenizer = AutoTokenizer.from_pretrained('lingxi', trust_remote_code=True)
    
