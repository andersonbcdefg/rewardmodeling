# Train a DeBERTa reward model on just the Anthropic hh-rlhf dataset.
import os
import warnings
warnings.simplefilter("ignore")
from dataclasses import dataclass
from functools import partial

import fire
import ray
import torch
from datasets import load_dataset
from accelerate import Accelerator
from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.accelerate import AccelerateTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import process_anthropic, tokenize_function


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

@dataclass
class Config: 
    num_workers: int = 10
    effective_batch_size: int =  80
    microbatch_size: int = 2
    max_lr: float = 3.0e-5
    grad_clip: float = None
    num_epochs: int = 4
    scheduler_steps: int = None

# Define your train worker loop
# config needs: effective_batch_size, microbatch_size, max_lr, grad_clip, scheduler_steps, num_epochs
def train_loop_per_worker(config):

    # Initialize the Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=(config.effective_batch_size // config.microbatch_size) // config.num_workers,
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Iterate over epochs and batches
    for epoch in range(config.num_epochs):
        for index, batch in enumerate(dataset_shard.iter_torch_batches(
            batch_size=config.microbatch_size, dtypes=torch.long
        )):
            input_ids, attn_mask = concat_batch(batch)
            with accelerator.accumulate(model):
                rewards = model(input_ids, attention_mask=attn_mask).logits
                micro_batch_loss = loss_fn(rewards)
                accelerator.backward(micro_batch_loss)
                if config.grad_clip is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Log metrics, save every 5000 batches
                if index % 5000 == 0:
                    session.report({
                        "loss": micro_batch_loss.item(), 
                        "epoch": epoch,
                        "batch": index,
                    },
                    checkpoint=Checkpoint.from_dict(
                        dict(epoch=epoch, model=accelerator.unwrap_model(model).state_dict())
                    ))
                else:
                    session.report({
                        "loss": micro_batch_loss.item(), 
                        "epoch": epoch,
                        "batch": index,
                    })

def main():
    # ray.init(
    #     runtime_env={
    #         "pip": [
    #             "datasets",
    #             "accelerate>=0.16.0",
    #             "transformers>=4.26.0",
    #             "torch>=1.12.0",
    #             "wandb==0.13.10"
    #         ]
    #     }
    # )

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-base-tasksource-nli")
    tokenize_partial = partial(tokenize_function, tokenizer=tokenizer, max_len=1024)
    train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    train_dataset = train_dataset.map(
        process_anthropic, remove_columns=train_dataset.column_names
    # Remove any examples that will get excessively truncated
    ).filter(
        lambda example: len(example['prompt']) + len(example['preferred']) < 3200 and len(example['prompt']) + len(example['dispreferred']) < 3200
    )
    train_dataset = train_dataset.map(
        tokenize_partial, batched=True, batch_size=1000, remove_columns=train_dataset.column_names
    )
    num_samples = len(train_dataset)
    train_dataset = ray.data.from_huggingface(train_dataset)
    
    NUM_EPOCHS = 4
    NUM_WORKERS = 10
    MICROBATCH_SIZE = 2
    EFFECTIVE_BATCH_SIZE = 80
    config = Config(
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        microbatch_size=2,
        max_lr=3.0e-5,
        grad_clip=None,
        num_epochs=NUM_EPOCHS,
        scheduler_steps = NUM_EPOCHS * num_samples // NUM_WORKERS // MICROBATCH_SIZE, # i think this is right?
        num_workers=NUM_WORKERS,
    ) 

    


    # Define scaling and run configs
    scaling_config = ScalingConfig(
        num_workers=10, 
        use_gpu=True
    )
    run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

    trainer = AccelerateTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        accelerate_config={},
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_dataset},
    )

    result = trainer.fit()
    best_checkpoint_loss = result.metrics["loss"]
    print(f"Best checkpoint loss: {best_checkpoint_loss}")


if __name__ == "__main__":
    fire.Fire(main)
