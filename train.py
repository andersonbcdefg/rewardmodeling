import fire
import torch
from accelerate import Accelerator
from data import get_dataloaders
from transformers import AutoModelForSequenceClassification
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


def train(
    wandb_api_key=None,
    project_name="train_reward_model",
    model_name="sileod/deberta-v3-base-tasksource-nli",
    freeze_layers=0,
    max_lr=3.0e-5,
    short_grad_clip=None,
    long_grad_clip=None,
    effective_batch_size=64,
    short_microbatch_size=16,
    long_microbatch_size=4,
    save_every=1000,
):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=effective_batch_size // short_microbatch_size,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True
    )
    if freeze_layers > 0:
        for n, p in model.named_parameters():
            if "embeddings" in n:
                p.requires_grad = False
            else:
                layer_num = int(n.split(".")[3])
                if layer_num < freeze_layers:
                    p.requires_grad = False

    short_dataloader, long_dataloader, eval_dataloader = get_dataloaders(
        pretokenized=True,
        short_bsz=short_microbatch_size,
        long_bsz=long_microbatch_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, betas=(0.9, 0.95) # , fused=True (doesn't work in torch < 2.0.0)
    )
    # constant learning rate
    scheduler_kwargs = {
        "max_lr": max_lr,
        "total_steps": len(short_dataloader) + len(long_dataloader),
        "pct_start": 0.005,
        "div_factor": 10,
        "final_div_factor": 1,
        "anneal_strategy": "linear",
        "three_phase": False,
    }
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)
    (
        short_dataloader,
        long_dataloader,
        eval_dataloader,
        model,
        optimizer,
    ) = accelerator.prepare(
        short_dataloader, long_dataloader, eval_dataloader, model, optimizer
    )

    if accelerator.is_main_process:
        if wandb_api_key is not None:
            wandb.login(key=wandb_api_key)
        else:
            wandb.login()
    wandb.init(
        project=project_name,
        group="DDP",
        config={
            "model_name": model_name,
            # "model": model.config.__dict__,
            "freeze_layers": freeze_layers,
            "scheduler": scheduler_kwargs,
        },
    )

    model.train()
    if accelerator.is_main_process:
        print("Training begins, with short_bsz", short_microbatch_size, "and long_bsz", long_microbatch_size)
    total_tokens = 0
    for index, batch in enumerate(short_dataloader):
        input_ids, attn_mask = concat_batch(batch)
        total_tokens += input_ids.numel()
        with accelerator.accumulate(model):
            rewards = model(input_ids, attention_mask=attn_mask).logits
            micro_batch_loss = loss_fn(rewards)
            wandb.log(
                {
                    "micro_batch_loss": micro_batch_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens": total_tokens,
                }
            )
            accelerator.backward(micro_batch_loss)
            if short_grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), short_grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if index % save_every == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir="checkpoints")

    # update gradient accumulation steps before increasing sequence length
    print("Increasing sequence length to 2048!")
    accelerator.wait_for_everyone()
    accelerator.gradient_accumulation_steps = (
        effective_batch_size // long_microbatch_size
    )
    for index, batch in enumerate(long_dataloader):
        input_ids, attn_mask = concat_batch(batch)
        total_tokens += input_ids.numel()
        with accelerator.accumulate(model):
            rewards = model(input_ids, attention_mask=attn_mask).logits
            micro_batch_loss = loss_fn(rewards)
            wandb.log(
                {
                    "micro_batch_loss": micro_batch_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens": total_tokens,
                }
            )
            accelerator.backward(micro_batch_loss)
            if long_grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), long_grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if index % save_every == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir="checkpoints")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Training complete. Saving final model...")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "final_model.pt")

if __name__ == "__main__":
    fire.Fire(train)
