import fire
import torch
import numpy as np
from tqdm import tqdm
from itertools import islice
from data import get_dataloaders
from train import concat_batch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def evaluate(ckpt_path, sample_every=10):
    # Setup model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "sileod/deberta-v3-base-tasksource-nli", 
        num_labels=1, 
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(ckpt_path))
    model.gradient_checkpointing_enable()
    
    # Get eval dataloader
    short_dataloader, long_dataloader, eval_dataloader = get_dataloaders(
        pretokenized=True,
        short_bsz=8,
        long_bsz=8
    )
    
    # Subsample for efficiency
    eval_subset = eval_subset = islice(eval_dataloader, sample_every - 1, None, sample_every)
    
    # Evaluate
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(eval_subset, total=len(eval_dataloader) // sample_every)):
            input_ids, attn_mask = concat_batch(batch)
            pref_rewards, dispref_rewards = model(input_ids.to(device), attention_mask=attn_mask.to(device)).logits.chunk(2, dim=0)
            correct = (pref_rewards.view(-1) > dispref_rewards.view(-1)).long()
            results.extend(list(correct.cpu().numpy()))
    
    print("Accuracy:", np.mean(results))
if __name__ == "__main__":
    fire.Fire(evaluate)