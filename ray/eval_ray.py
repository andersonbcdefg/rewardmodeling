import numpy as np
from functools import partial
import ray
import ray.data
from ray.air import Checkpoint
from datasets import load_dataset
from data import tokenize_function, process_anthropic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class PredictCallable:
    def __init__(self, model_id: str, state_dict: dict = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            num_labels=1, 
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, batch: dict) -> list:
        pref_ids, pref_mask, dispref_ids, dispref_mask = (
            batch['preferred_input_ids'].to(self.model.device), 
            batch['preferred_attention_masks'].to(self.model.device),
            batch['dispreferred_input_ids'].to(self.model.device),
            batch['dispreferred_attention_masks'].to(self.model.device)
        )
        pref_rewards = self.model(pref_ids, attention_mask=pref_mask).logits.view(-1)
        dispref_rewards = self.model(dispref_ids, attention_mask=dispref_mask).logits.view(-1)
        correct = (pref_rewards > dispref_rewards).long()
        
        return list(correct.cpu().numpy())

if __name__ == "__main__":
    context = ray.init()
    print(ray.cluster_resources())

    directory_path = "/mnt/cluster_storage/AccelerateTrainer_2023-05-28_23-32-46/AccelerateTrainer_9feab_00000_0_2023-05-28_23-32-48/checkpoint_000007"
    ckpt = Checkpoint.from_directory(directory_path).to_dict()
    state_dict = ckpt['model']

    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test").select(range(1000))
    eval_dataset = eval_dataset.map(
        process_anthropic, remove_columns=eval_dataset.column_names
    )
    eval_dataset_tokenized = eval_dataset.map(partial(tokenize_function, tokenizer=tokenizer, max_len=1024), 
                                        batched=True, remove_columns=eval_dataset.column_names)
    ray_eval_dataset = ray.data.from_huggingface(eval_dataset_tokenized)

    preds = (
        ray_eval_dataset
        .repartition(100)
        .map_batches(
            PredictCallable,
            batch_size=4,
            fn_constructor_kwargs=dict(model_id=MODEL_NAME, state_dict=state_dict),
            compute=ray.data.ActorPoolStrategy(10, 10),
            num_gpus=1,
        )
    )

    preds.take_all()