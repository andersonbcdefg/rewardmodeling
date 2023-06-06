import os
import fire
import torch
import numpy as np
from accelerate import Accelerator
from data import get_train_dataloader, get_eval_dataloaders
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import wandb


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

def evaluate(accelerator, model, eval_dataloaders):
    metrics = {}
    model.eval()
    with torch.no_grad():
        for key in eval_dataloaders:
            loader = eval_dataloaders[key]
            correct = []
            losses = []
            for batch in loader:
                input_ids, attn_mask = concat_batch(batch)
                pref_rewards, dispref_rewards = model(input_ids, attention_mask=attn_mask).logits.chunk(2, dim=0)
                batch_correct = (pref_rewards.view(-1) > dispref_rewards.view(-1)).long()
                batch_loss = loss_fn(pref_rewards, dispref_rewards)
                correct.extend(accelerator.gather(batch_correct).cpu().numpy())
                losses.extend(accelerator.gather(batch_loss).cpu().numpy())
            correct = correct[:len(loader.dataset)]
            losses = losses[:len(loader.dataset)]
            metrics[f"{key}_accuracy"] = np.mean(correct)
            metrics[f"{key}_loss"] = np.mean(losses)

    return metrics

# before running training, you need to prepare data by running data.py and saving dataloaders to disk
def train(
    wandb_api_key=None,
    project_name="train_reward_model",
    # group_name="DDP",
    model_name="sileod/deberta-v3-base-tasksource-nli",
    gradient_checkpointing=False,
    freeze_layers=0,
    num_epochs=5,
    max_lr=3.0e-5,
    grad_clip=None,
    effective_batch_size=64,
    microbatch_size=16,
    # save_every=1000,
    data_dir="data",
    save_dir="checkpoints",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    accumulation_steps = effective_batch_size // microbatch_size
    
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=accumulation_steps
    )

    # Then load dataset from the disk for all processes
    train_dataloader = torch.load(os.path.join(data_dir, "train_dataloader.pt"))
    datasets = train_dataloader["datasets"]
    train_dataloader = train_dataloader["dataloader"]
    eval_dataloaders = torch.load(os.path.join(data_dir, "eval_dataloaders.pt"))
    
    with accelerator.main_process_first():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if freeze_layers > 0:
        for n, p in model.named_parameters():
            if "embeddings" in n:
                p.requires_grad = False
            else:
                layer_num = int(n.split(".")[3])
                if layer_num < freeze_layers:
                    p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, betas=(0.9, 0.95) # , fused=True (doesn't work in torch < 2.0.0)
    )

    # constant learning rate
    scheduler_kwargs = {
        "max_lr": max_lr,
        "total_steps": len(train_dataloader) * num_epochs,
        "pct_start": 0.005,
        "div_factor": 10,
        "final_div_factor": 1,
        "anneal_strategy": "linear",
        "three_phase": False,
    }
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)
    (
        train_dataloader,
        model,
        optimizer,
    ) = accelerator.prepare(
        train_dataloader, model, optimizer
    )

    for key in eval_dataloaders:
        eval_dataloaders[key] = accelerator.prepare_data_loader(eval_dataloaders[key])

    if accelerator.is_main_process:
        if wandb_api_key is not None:
            wandb.login(key=wandb_api_key)
        else:
            wandb.login()
        wandb.init(
            project=project_name,
            config={
                "model_name": model_name,
                # "model": model.config.__dict__,
                "freeze_layers": freeze_layers,
                "scheduler": scheduler_kwargs,
                "datasets": datasets,
            },
        )

    model.train()
    if accelerator.is_main_process:
        print("Training begins!")
    total_tokens = 0
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids, attn_mask = concat_batch(batch)
            total_tokens += input_ids.numel()
            with accelerator.accumulate(model):
                rewards = model(input_ids, attention_mask=attn_mask).logits
                micro_batch_loss = loss_fn(rewards)
                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "micro_batch_loss": micro_batch_loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "tokens": total_tokens,
                        }
                    )
                accelerator.backward(micro_batch_loss)
                if grad_clip is not None:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
        
        # Evaluate & save model at the end of each epoch
        print("Evaluating...")
        metrics = evaluate(accelerator, model, eval_dataloaders)
        if accelerator.is_main_process:
            wandb.log(metrics)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("Saving model...")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"model_{epoch}.pt"))
            print("Model saved!")

if __name__ == "__main__":
    fire.Fire(train)
