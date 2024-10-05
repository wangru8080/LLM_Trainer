import sys
import os
import torch
import transformers
import datasets
from transformers import (
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AddedToken
)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOTrainer, ORPOTrainer
import glob
import argparse
import logging
import yaml
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from arguments import ModelArguments, DataTrainingArguments, ExtraTrainingArguments
from build_dataset import build_sft_dataset, build_pretrain_dataset, build_dpo_dataset, DataCollatorForPadding
from trainer import CustomizedTrainer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_args_file', type=str, default=None)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    train_args_file = args.train_args_file
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtraTrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=train_args_file)

    if training_args.local_rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        train_args = yaml.safe_load(open(train_args_file))
        yaml.dump(train_args, open(os.path.join(training_args.output_dir, 'train_args.yaml'), 'w'))
    
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info('train_args:{}'.format(training_args))

    return model_args, data_args, training_args

def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names

def load_model(model_args, training_args):
    config_kwargs = dict(
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=model_args.trust_remote_code,
        _attn_implementation='flash_attention_2' if training_args.use_flash_att else 'sdpa'
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else model_args.use_fast,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)

    # 部分模型的base与chat版本的tokenizer存在差异
    if 'internlm2' in model_args.model_name_or_path.lower():
        tokenizer._added_tokens_encoder.update({'<|im_start|>': 92543})
        tokenizer._added_tokens_encoder.update({'<|im_end|>': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('<|im_start|>')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('<|im_end|>')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    elif 'orion' in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    elif 'gemma' in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})
    
    if training_args.task_type in ['dpo', 'orpo'] and 'Qwen' in tokenizer.__class__.__name__: # qwen没有bos_token，要设置一下，不然dpo train时会报错
        tokenizer.add_special_tokens(dict(bos_token=tokenizer.eos_token))
        tokenizer.bos_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=tokenizer.eos_token))
        tokenizer.pad_token_id = tokenizer.eos_token_id

    assert tokenizer.pad_token_id is not None, 'pad_token_id should not be None'
    assert tokenizer.eos_token_id is not None, 'eos_token_id should not be None'
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')

    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {model_args.model_name_or_path}')
    logger.info(f'Train model with {training_args.train_mode}')

    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if training_args.train_mode == 'qlora':
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            llm_int8_skip_modules = training_args.modules_to_save.split(',')
        else:
            llm_int8_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            llm_int8_skip_modules=llm_int8_skip_modules,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        quantization_config = None
    if quantization_config is not None:
        logger.info(f'quantization_config:{quantization_config.to_dict()}')
    
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    ddp = world_size != 1
    if ddp:
        device_map = {'': int(os.environ.get('LOCAL_RANK', '0'))}
    else:
        device_map = 'auto'
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config
    )
    if training_args.task_type == 'zero-pt':
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **model_kwargs)

    if training_args.train_mode == 'qlora' and training_args.task_type in ['pt', 'sft']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    if training_args.task_type in ['pt', 'zero-pt', 'sft']:
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.train_mode == 'full':
        peft_config = None
    else:
        logger.info('Init peft model')
        target_modules = find_all_linear_names(model, training_args.train_mode)
        modules_to_save = training_args.modules_to_save
        if modules_to_save:
            modules_to_save = modules_to_save.split(',')
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        logger.info(f'lora_rank: {lora_rank}')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            r=lora_rank, 
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save
        )

    if training_args.train_mode in ['lora', 'qlora'] and training_args.task_type in ['pt', 'zero-pt', 'sft']:
        model = get_peft_model(model, peft_config)

        if training_args.trainable_params:
            # enable trainable params
            [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(',')])]

        logger.info(f'model.modules_to_save: {model.modules_to_save}')
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()
    
    # init ref_model
    if training_args.task_type == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs) if training_args.train_mode == 'full' else None
    # pretrain和sft，不需要ref_model
    else:
        ref_model = None
    
    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fB" % (total / 1e9))

    return {
        'model': model,
        'tokenizer': tokenizer,
        'ref_model': ref_model,
        'peft_config': peft_config
    }

def load_dataset(data_args, training_args, tokenizer):
    train_dataset = None
    eval_dataset=None
    if training_args.do_train:
        with training_args.main_process_first(local=False, desc='loading and tokenization'):
            if data_args.train_file:
                files = [data_args.train_file]
            else:
                files = glob.glob(data_args.train_file_dir + '/*.json*')
            logger.info(f'training files: {" ".join(files)}')

            if training_args.task_type in ['pt', 'zero-pt']:
                train_dataset = build_pretrain_dataset(
                    data_path=files,
                    tokenizer=tokenizer,
                    max_seq_len=data_args.max_seq_length,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers
                )
            elif training_args.task_type == 'sft':
                train_dataset = build_sft_dataset(
                    data_path=files,
                    tokenizer=tokenizer,
                    max_seq_len=data_args.max_seq_length,
                    discard_long_sample=data_args.discard_long_sample,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers,
                    template_name=data_args.template_name
                )
            elif training_args.task_type in ['dpo', 'orpo']:
                train_dataset = build_dpo_dataset(
                    data_path=files,
                    max_seq_len=data_args.max_seq_length,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers,
                    template_name=data_args.template_name,
                )
            else:
                pass
    
    if training_args.do_eval:
        with training_args.main_process_first(local=False, desc='loading and tokenization'):
            if data_args.train_file:
                files = [data_args.eval_file]
            else:
                files = glob.glob(data_args.eval_file_dir + '/*.json*')
            logger.info(f'evaluation files: {" ".join(files)}')

            if training_args.task_type in ['pt', 'zero-pt']:
                eval_dataset = build_pretrain_dataset(
                    data_path=files,
                    tokenizer=tokenizer,
                    max_seq_len=data_args.max_seq_length,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers
                )
            elif training_args.task_type == 'sft':
                eval_dataset = build_sft_dataset(
                    data_path=files,
                    tokenizer=tokenizer,
                    max_seq_len=data_args.max_seq_length,
                    discard_long_sample=data_args.discard_long_sample,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers,
                    template_name=data_args.template_name
                )
            elif training_args.task_type in ['dpo', 'orpo']:
                eval_dataset = build_dpo_dataset(
                    data_path=files,
                    max_seq_len=data_args.max_seq_length,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers,
                    template_name=data_args.template_name,
                )
            else:
                pass
        
        logger.info(f'Num train_samples {len(train_dataset)}Num train_samples {len(train_dataset)}')
        if training_args.task_type in ['dpo', 'orpo']:
            logger.info(f'training example:\n{train_dataset[0]["prompt"] + train_dataset[0]["chosen"]}')
        else:
            logger.info(f'training example:\n{tokenizer.decode(train_dataset[0]["input_ids"])}')
    
    return train_dataset, eval_dataset

def main():
    model_args, data_args, training_args = setup_everything()

    components = load_model(model_args, training_args)
    model = components['model']
    tokenizer = components['tokenizer']
    ref_model = components['ref_model']
    peft_config = components['peft_config']

    train_dataset, eval_dataset = load_dataset(data_args, training_args, tokenizer)

    if training_args.task_type == 'dpo':
        trainer = DPOTrainer(
            model,
            ref_model=ref_model,
            args=training_args,
            beta=training_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_prompt_length=training_args.max_prompt_length,
            max_length=data_args.max_seq_length
        )
    elif training_args.task_type == 'orpo':
        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    else:
        # pretrain or sft
        data_collator = DataCollatorForPadding(tokenizer=tokenizer)
        trainer = CustomizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Training
    if training_args.do_train:
        # 开始训练
        logger.info('*** starting training ***')
        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            logger.info('*** starting last_checkpoint training ***')
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                    'Use --overwrite_output_dir to overcome.'
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                    'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
                )
        
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info('*** Save model ***')
        trainer.save_model(training_args.output_dir)
        logger.info(f'Model saved to {training_args.output_dir}')
        # 保存训练指标
        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(eval_dataset)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)


if __name__ == '__main__':
    main()
