import os
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer


def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, checkpoint_dir: str, accelerator: Accelerator):
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(checkpoint_dir)

    accelerator.wait_for_everyone()