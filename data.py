import re
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets

def process_shp(example):
    if example['labels'] == 1:
        preferred = example['human_ref_A']
        dispreferred = example['human_ref_B']
    else:
        preferred = example['human_ref_B']
        dispreferred = example['human_ref_A']
    return {
        "prompt": example["history"],
        "preferred": preferred,
        "dispreferred": dispreferred,
    }

def split_by_prefix(A, B):
    shared_prefix = ""
    for i in range(min(len(A), len(B))):
        if A[i] != B[i]:
            break
        shared_prefix += A[i]
        
    A_suffix = A[len(shared_prefix):]
    B_suffix = B[len(shared_prefix):]
    
    return (shared_prefix, A_suffix, B_suffix)

def split_by_assistant(s):
    split_string = "Assistant:"
    index = s.rfind(split_string)
    if index == -1:
        return (s, "")
    else:
        prefix = s[:index+len(split_string)]
        suffix = s[index+len(split_string):]
        return (prefix, suffix)


def process_anthropic(example):
    # find longest shared prefix
    shared_prefix, chosen_suffix, rejected_suffix = split_by_prefix(example["chosen"], example["rejected"])
    
    # split off the part of the prompt up to and including the last "Assistant:"
    prompt, shared_suffix = split_by_assistant(shared_prefix)

    return {
        "prompt": prompt,
        "preferred": shared_suffix + chosen_suffix,
        "dispreferred": shared_suffix + rejected_suffix,
    }

def process_webgpt(example):
    # Define the regex pattern
    pattern = r'\[\d{1,2}(?:,\s*\d{1,2})*\]'
    answer_1 = example['answer_1']
    answer_0 = example['answer_0']
    # print(answer_1, "||", answer_0)

    prompt = example['question']['full_text']
    if example['score_0'] > 0:
        preferred = answer_0
        dispreferred = answer_1
    elif example['score_1'] > 0:
        preferred = answer_1
        dispreferred = answer_0
    else:
        preferred = ""
        dispreferred = ""
    if not isinstance(preferred, str):
        print(preferred)
        # preferred = ""
    if not isinstance(dispreferred, str):
        print(dispreferred)
        # dispreferred = ""
    return {
        "prompt": prompt,
        "preferred": re.sub(pattern, '', preferred),
        "dispreferred": re.sub(pattern, '', dispreferred)
    }

def get_combined_datasets():
    # process all datasets to have the same 3 columns: prompt, preferred, dispreferred
    shp_dataset = load_dataset("stanfordnlp/SHP", split="train")
    shp_dataset = shp_dataset.map(
        process_shp, remove_columns=shp_dataset.column_names
    )

    hh_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    hh_dataset = hh_dataset.map(
        process_anthropic, remove_columns=hh_dataset.column_names
    )

    webgpt_dataset = load_dataset("openai/webgpt_comparisons", split="train")
    webgpt_dataset = webgpt_dataset.map(
        process_webgpt, remove_columns=webgpt_dataset.column_names
    ).filter(
        lambda example: example["preferred"] != "" and example["dispreferred"] != ""
    )

    # total_samples = len(shp_dataset) + len(hh_dataset) + len(webgpt_dataset)
    
    combined = concatenate_datasets([shp_dataset, hh_dataset, webgpt_dataset]).shuffle(seed=42).flatten_indices()
    combined = combined.map(
        lambda example: {
            "length": len(example["prompt"]) + len(example["preferred"]) + len(example["dispreferred"]),
        }
    ).filter(
        lambda example: example["length"] < 10000
    )

    median = sorted(combined["length"])[len(combined) // 2]
    print("median length:", median)

    long = combined.filter(lambda example: example["length"] > 1250)
    short = combined.filter(lambda example: example["length"] <= 1250)

    return {
        "long": long,
        "short": short,
    }

def tokenize_function(examples, tokenizer, max_len):
    preferred_input_ids = torch.full((len(examples), max_len), tokenizer.pad_token_id, dtype=torch.long)
    dispreferred_input_ids = torch.full((len(examples), max_len), tokenizer.pad_token_id, dtype=torch.long)
    preferred_attention_masks = torch.zeros((len(examples), max_len), dtype=torch.long)
    dispreferred_attention_masks = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, example in enumerate(examples):
        prompt_tokenized = tokenizer.encode(example["prompt"], add_special_tokens=True, return_tensors="pt").view(-1) # prompt will have CLS and end with SEP
        
        preferred_tokenized = tokenizer.encode(example["preferred"], add_special_tokens=False, return_tensors="pt").view(-1) # response should not have CLS, it continues the prompt
        prompt_with_preferred = torch.cat((prompt_tokenized, preferred_tokenized))
        preferred_mask = torch.cat((torch.ones(len(prompt_with_preferred)), torch.zeros(max_len - len(prompt_with_preferred))))
        preferred_input_ids[i, :] = prompt_with_preferred[:max_len]
        preferred_attention_masks[i, :] = preferred_mask[:max_len]

        dispreferred_tokenized = tokenizer.encode(example["dispreferred"], add_special_tokens=False, return_tensors="pt").view(-1) 
        prompt_with_dispreferred = torch.cat((prompt_tokenized, dispreferred_tokenized))
        dispreferred_mask = torch.cat((torch.ones(len(prompt_with_dispreferred)), torch.zeros(max_len - len(prompt_with_dispreferred))))
        dispreferred_input_ids[i, :] = prompt_with_dispreferred[:max_len]
        dispreferred_attention_masks[i, :] = dispreferred_mask[:max_len]

    return {
        "preferred_input_ids": preferred_input_ids,
        "preferred_attention_masks": preferred_attention_masks,
        "dispreferred_input_ids": dispreferred_input_ids,
        "dispreferred_attention_masks": dispreferred_attention_masks
    }  

# adapted from https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2
# for this implementation, assume the hf_dataset is already tokenized!
class PairwiseDataset(torch.utils.data.Dataset):
     def __init__(self, hf_dataset):
         self.chosen_input_ids = []
         self.chosen_attn_masks = []
         self.rejected_input_ids = []
         self.rejected_attn_masks = []
         for pair in pairs:
             chosen, rejected = pair["chosen"], pair["rejected"]
             chosen_encodings_dict = tokenizer(
                 "<|startoftext|>" + chosen + "<|endoftext|>",
                 truncation=True,
                 max_length=max_length,
                 padding="max_length",
                 return_tensors="pt",
             )
             rejected_encodings_dict = tokenizer(
                 "<|startoftext|>" + rejected + "<|endoftext|>",
                 truncation=True,
                 max_length=max_length,
                 padding="max_length",
                 return_tensors="pt",
             )
             self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
             self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
             self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
             self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])


     def __len__(self):
         return len(self.chosen_input_ids)


     def __getitem__(self, idx):
         return (
             self.chosen_input_ids[idx],
             self.chosen_attn_masks[idx],
             self.rejected_input_ids[idx],
             self.rejected_attn_masks[idx],
         )
