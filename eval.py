import fire
import torch
import numpy as np
from tqdm import tqdm
from itertools import islice
from data import get_eval_dataloaders
from train import concat_batch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def steamshp_eval_loop(model, dataloader, device):
    pass


def eval_loop(model, loader, n_batches, device):
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, total=n_batches, desc="Evaluating..."):
            input_ids, attn_mask = concat_batch(batch)
            pref_rewards, dispref_rewards = model(
                input_ids.to(device), attention_mask=attn_mask.to(device)
            ).logits.chunk(2, dim=0)
            correct = (pref_rewards.view(-1) > dispref_rewards.view(-1)).long()
            results.extend(list(correct.cpu().numpy()))
    return np.mean(results)


def evaluate(
    model_name="sileod/deberta-v3-base-tasksource-nli",
    ckpt_path=None,
    sample_every=10,
    bsz=16,
    max_len=2048,
):
    # Setup model and tokenizer
    if "deberta-v3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
    elif "SteamSHP" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    else:
        raise ValueError("Model name not recognized")

    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    model.gradient_checkpointing_enable()

    # Get eval dataloaders
    dataloaders = get_eval_dataloaders(
        tokenizer, bsz, max_len, steamshp=("SteamSHP" in model_name)
    )

    # Subsample for efficiency & evaluate!
    results = {}
    for name, loader in dataloaders.items():
        subset = islice(loader, sample_every - 1, None, sample_every)
        n_batches = len(loader) // sample_every
        result = eval_loop(model, subset, n_batches, torch.device("cuda"))
        results[name] = result

    for name, result in results.items():
        print(f"{name}: {result:.4f}")


if __name__ == "__main__":
    fire.Fire(evaluate)
