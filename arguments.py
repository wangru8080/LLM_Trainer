from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from trl import DPOConfig, ORPOConfig
from template import template_dict

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a json file)."})
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input json data file folder."})
    eval_file: Optional[str] = field(default=None, metadata={"help": "The input evaluate data file (a json file)."})
    eval_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation json file folder."})
    validation_split_percentage: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The cache for training dataset"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "[Finetuning] Maximum length of sequences."})
    template_name: str = field(default="qwen", metadata={"help": "data template", "choices": list(template_dict.keys())})
    discard_long_sample: bool = field(
        default=True,
        metadata={"help": "the parameter is used to indicate whether to perform truncation or discard"
                          "operations when the sample is too long. If set to False, it indicates truncation; "
                          "If set to True, it indicates discard."}
    )

@dataclass
class ExtraTrainingArguments(DPOConfig, ORPOConfig, TrainingArguments):
    task_type: str = field(default="sft", metadata={"help": "[zero-pt, pt, sft, dpo, orpo]", "choices": ["zero-pt", "pt", "sft", "dpo", "orpo"]})
    train_mode: str = field(default="qlora", metadata={"help": "[full, lora, qlora]", "choices": ["full", "lora", "qlora"]})
    lora_rank : Optional[int] = field(default=None)
    lora_dropout : Optional[float] = field(default=None)
    lora_alpha : Optional[float] = field(default=None)
    modules_to_save : Optional[str] = field(default=None, metadata={"choices": [None, "embed_tokens,lm_head"]})
    trainable_params: Optional[str] = field(
        default=None,
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training.", "choices": [None, "embed,norm"]},
    )
    use_flash_att: bool = field(
        default=False,
        metadata={"help": "Use flash attention"}
    )
    load_in_kbits: Optional[int] = field(default=None, metadata={"help": "Wheter to use qlora", "choices": [None, 8, 4]})
    use_flash_attn_ce_loss: Optional[bool] = field(default=False, metadata={"help": "Whether to use CrossEntropyLoss from flash-attention"})
    train_shuffle: Optional[bool] = field(default=False, metadata={"help": "Whether to train shuffle"})
