# Train a DeBERTa reward model on just the Anthropic hh-rlhf dataset.
import os
from collections import namedtuple

import fire
import ray
import torch
from datasets import load_dataset
from accelerate import Accelerator
from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.accelerate import AccelerateTrainer
from transformers import AutoModelForSequenceClassification

from data import process_anthropic


# Calculate loss on rewards (first half should be preferred, second half should be dispreferred)
def loss_fn(rewards):
    pref_rewards, dispref_rewards = rewards.chunk(2, dim=0)
    bsz = pref_rewards.numel()
    return (
        -torch.log(
            torch.sigmoid(pref_rewards.view(-1) - dispref_rewards.view(-1))
        ).sum()
        / bsz
    )


# Combine preferred and dispreferred sequences into 1 batch
def concat_batch(batch):
    pref_ids, pref_mask, dispref_ids, dispref_mask = (
        batch["preferred_input_ids"],
        batch["preferred_attention_masks"],
        batch["dispreferred_input_ids"],
        batch["dispreferred_attention_masks"],
    )
    input_ids, attn_mask = torch.cat([pref_ids, dispref_ids], dim=0), torch.cat(
        [pref_mask, dispref_mask], dim=0
    )
    return input_ids, attn_mask

Config = namedtuple("Config", ["effective_batch_size", "microbatch_size", "max_lr", "grad_clip", "num_epochs"])
config = Config(
    effective_batch_size=64,
    microbatch_size=16,
    max_lr=3.0e-5,
    grad_clip=None,
    num_epochs=4,
)

# Define your train worker loop
# config needs: effective_batch_size, microbatch_size, max_lr, grad_clip, scheduler_steps, num_epochs
def train_loop_per_worker(config):

    # Initialize the Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.effective_batch_size // config.microbatch_size,
    )

    # Fetch training set from the session
    dataset_shard = session.get_dataset_shard("train")

    # Loss function, optimizer, prepare model for training.
    # This moves the data and prepares model for distributed
    # execution
    model = AutoModelForSequenceClassification.from_pretrained(
        "sileod/deberta-v3-base-tasksource-nli", num_labels=1, ignore_mismatched_sizes=True
    )
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(0.9, 0.95),  # , fused=True (doesn't work in torch < 2.0.0)
    )
    # constant learning rate
    scheduler_kwargs = {
        "max_lr": config.max_lr,
        "total_steps": config.scheduler_steps, # or get the length of the shard?
        "pct_start": 0.01,
        "div_factor": 10,
        "final_div_factor": 1,
        "anneal_strategy": "linear",
        "three_phase": False,
    }
    scheduler = torch.optim.lr_scheduler.OneCycleLR(**scheduler_kwargs)
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Iterate over epochs and batches
    for epoch in range(config.num_epochs):
        for index, batch in enumerate(dataset_shard.iter_torch_batches(
            batch_size=32, dtypes=torch.float
        )):
            input_ids, attn_mask = concat_batch(batch):
            with accelerator.accumulate(model):
                rewards = model(input_ids, attention_mask=attn_mask).logits
                micro_batch_loss = loss_fn(rewards)
                accelerator.backward(micro_batch_loss)
                if config.grad_clip is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Log metrics
                session.report({
                        "loss": micro_batch_loss.item(), 
                        "epoch": epoch,
                        "batch": index,
                    },
                    checkpoint=Checkpoint.from_dict(
                        dict(epoch=epoch, model=accelerator.unwrap_model(model).state_dict(),
                    )
                ),
        )
        
torch.manual_seed(42)

train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
train_dataset = train_dataset.map(
    process_anthropic, remove_columns=hh_dataset.column_names
)

# Define scaling and run configs
scaling_config = ScalingConfig(
    num_workers=10, 
    use_gpu=True
)
run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

trainer = AccelerateTrainer(
    train_loop_per_worker=train_loop_per_worker,
    # Instead of using a dict, you can run ``accelerate config``.
    # The default value of None will then load that configuration
    # file.
    accelerate_config={},
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": train_dataset},
)

result = trainer.fit()

best_checkpoint_loss = result.metrics["loss"]

    

    


  

    

if __name__ == "__main__":
    fire.Fire(train)
