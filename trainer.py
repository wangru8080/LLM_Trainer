import logging
import torch
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import SequentialSampler
from typing import Optional, Optional
import datasets
from transformers import (
    Trainer,
    is_datasets_available
)
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
    RandomSampler
)
from transformers.utils import (
    is_peft_available
)
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_utils import has_length
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import PeftModel
from packaging import version
import importlib.metadata
from flash_attn.losses.cross_entropy import CrossEntropyLoss

logger = logging.getLogger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version('peft')) >= version.parse('0.7.0'):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

class CustomizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.train_shuffle:
            # Build the sampler.
            if self.args.group_by_length:
                if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                    lengths = (
                        self.train_dataset[self.args.length_column_name]
                        if self.args.length_column_name in self.train_dataset.column_names
                        else None
                    )
                else:
                    lengths = None
                model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )

            else:
                return RandomSampler(self.train_dataset)
        else:
            return SequentialSampler(self.train_dataset)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None and "labels" in inputs) or self.args.use_flash_attn_ce_loss:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if self.args.use_flash_attn_ce_loss:
                logits = outputs["logits"]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(inplace_backward=True)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                outputs['loss'] = loss
            else:
                unwrapped_model = unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
