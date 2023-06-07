import os
import re
import fire
from functools import partial
import torch
from datasets import load_dataset, interleave_datasets
from transformers import DefaultDataCollator, AutoTokenizer

def process_shp(example):
    if example["labels"] == 1:
        preferred = example["human_ref_A"]
        dispreferred = example["human_ref_B"]
    else:
        preferred = example["human_ref_B"]
        dispreferred = example["human_ref_A"]
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

    A_suffix = A[len(shared_prefix) :]
    B_suffix = B[len(shared_prefix) :]

    return (shared_prefix, A_suffix, B_suffix)


def split_by_assistant(s):
    split_string = "Assistant:"
    index = s.rfind(split_string)
    if index == -1:
        return (s, "")
    else:
        prefix = s[: index + len(split_string)]
        suffix = s[index + len(split_string) :]
        return (prefix, suffix)


def process_anthropic(example):
    # find longest shared prefix
    shared_prefix, chosen_suffix, rejected_suffix = split_by_prefix(
        example["chosen"], example["rejected"]
    )

    # split off the part of the prompt up to and including the last "Assistant:"
    prompt, shared_suffix = split_by_assistant(shared_prefix)

    return {
        "prompt": prompt.strip(),
        "preferred": shared_suffix + chosen_suffix,
        "dispreferred": shared_suffix + rejected_suffix,
    }

def normalize_whitespace(s):
    s = s.replace(".  ", ". ")
    lines = s.split("\n")
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]
    return "\n".join(lines)

def process_gptj(example):
    return {
        "prompt": example["prompt"],
        "preferred": normalize_whitespace(example["chosen"]),
        "dispreferred": normalize_whitespace(example["rejected"]),
    }

def process_webgpt(example):
    # Define the regex pattern
    pattern = r"\[\d{1,2}(?:,\s*\d{1,2})*\]"
    answer_1 = example["answer_1"]
    answer_0 = example["answer_0"]
    # print(answer_1, "||", answer_0)

    prompt = example["question"]["full_text"]
    if example["score_0"] > 0:
        preferred = answer_0
        dispreferred = answer_1
    elif example["score_1"] > 0:
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
        "preferred": re.sub(pattern, "", preferred).strip(),
        "dispreferred": re.sub(pattern, "", dispreferred).strip(),
    }


def process_alpaca(example):
    prompt = example["instruction"]
    if example["input"] != "":
        prompt += "\nInput: "
        prompt += example["input"]
    if example["preference"] == 1:
        preferred = example["output_1"]
        dispreferred = example["output_2"]
    else:
        preferred = example["output_2"]
        dispreferred = example["output_1"]
    return {"prompt": prompt, "preferred": preferred, "dispreferred": dispreferred}

def process_synthetic(example):
    return {
        "prompt": example["prompt"],
        "preferred": example['response_a'] if example['preferred'] == "A" else example['response_b'],
        "dispreferred": example['response_b'] if example['preferred'] == "A" else example['response_a']
    }

def add_human_and_assistant_to_prompt(prompt):
    if not prompt.startswith("Human:"):
        prompt = "Human: " + prompt
    if not prompt.endswith("Assistant:"):
        prompt = prompt.strip() + "\n\nAssistant:"
    return prompt

TRAIN_DATASETS = {
    "hh": {
        "hub_url": "Anthropic/hh-rlhf",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_anthropic,
    },
    "shp": {
        "hub_url": "stanfordnlp/SHP",
        "split": "train",
        "filter_fn": lambda example: example["score_ratio"] >= 2.0 and example["score_ratio"] <= 8.0,
        "processing_fn": process_shp,
    },
    "webgpt": {
        "hub_url": "openai/webgpt_comparisons",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_webgpt,
    },
    "summarize": {
        "hub_url": "openai/summarize_from_feedback",
        "split": "train",
        "filter_fn": None,
        "processing_fn": None,
    },
    "synth_gptj": {
        "hub_url": "Dahoas/synthetic-instruct-gptj-pairwise",
        "split": "train",
        "filter_fn": None, 
        "processing_fn": process_gptj,
    },
    "synth_dolly": {
        "hub_url": "andersonbcdefg/dolly_reward_modeling_pairwise",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_synthetic,
    },
    "synth_redteam": {
        "hub_url": "andersonbcdefg/red_teaming_reward_modeling_pairwise_no_as_an_ai",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_synthetic,
    },
    "synth_gpteacher": {
        "hub_url": "andersonbcdefg/gpteacher_reward_modeling_pairwise",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_synthetic,
    },
    "synth_sharegpt": {
        "hub_url": "andersonbcdefg/sharegpt_reward_modeling_pairwise_no_as_an_ai",
        "split": "train",
        "filter_fn": None,
        "processing_fn": process_synthetic,
    },
}

EVAL_DATASETS = {
    "hh": {
        "hub_url": "Anthropic/hh-rlhf",
        "split": "test",
        "filter_fn": None,
        "processing_fn": process_anthropic,
    },
    "shp": {
        "hub_url": "stanfordnlp/SHP",
        "split": "test",
        "filter_fn": None,
        "processing_fn": process_shp,
    },
    "alpaca_gpt4": {
        "hub_url": "tatsu-lab/alpaca_farm",
        "subset": "alpaca_gpt4_preference",
        "split": "preference",
        "filter_fn": None,
        "processing_fn": process_alpaca,
    },
    "alpaca_human": {
        "hub_url": "tatsu-lab/alpaca_farm",
        "split": "preference",
        "subset": "alpaca_human_preference",
        "filter_fn": None,
        "processing_fn": process_alpaca,
    },
}

def get_datasets(
        datasets=["hh"], 
        add_human_assistant_labels=[],
        min_length_in_tokens=None, 
        max_length_in_tokens=None, 
        tokenizer=None,
        train=True
):
    if train:
        registry=TRAIN_DATASETS
    else:
        registry=EVAL_DATASETS

    if tokenizer is None:
        if min_length_in_tokens is not None or max_length_in_tokens is not None:
            raise ValueError("Cannot specify min_length_in_tokens or max_length_in_tokens without specifying tokenizer.")
    
    if datasets == "all":
        datasets = list(registry.keys())

    result = {}
    # process all datasets to have the same 3 columns: prompt, preferred, dispreferred
    for ds_name in datasets:
        if "subset" in registry[ds_name]:
            subset = registry[ds_name]["subset"]
            loaded = load_dataset(registry[ds_name]["hub_url"], subset, split=registry[ds_name]["split"])
        else:
            loaded = load_dataset(registry[ds_name]["hub_url"], split=registry[ds_name]["split"])
        if registry[ds_name]["filter_fn"] is not None:
            loaded = loaded.filter(registry[ds_name]["filter_fn"])
        if registry[ds_name]["processing_fn"] is not None:
            print(loaded[0].keys())
            loaded = loaded.map(registry[ds_name]["processing_fn"], remove_columns=loaded.column_names).filter(
                lambda example: example["preferred"] != "" and example["dispreferred"] != ""
            )
        result[ds_name] = loaded

    # add human and assistant to prompt if applicable
    for key in result.keys():
        if key in add_human_assistant_labels:
            result[key] = result[key].map(
                lambda example: {"prompt": add_human_and_assistant_to_prompt(example['prompt'])}
            )

    # filter all datasets for length
    if min_length_in_tokens is not None or max_length_in_tokens is not None:
        for key in result.keys():
            result[key] = result[key].map(
                lambda example: {"length": len(tokenizer(example['prompt']).input_ids) + \
                                max(len(tokenizer(example['preferred']).input_ids), 
                                    len(tokenizer(example['dispreferred']).input_ids))},
            )
            if min_length_in_tokens is not None:
                result[key] = result[key].filter(lambda example: example['length'] >= min_length_in_tokens)
            if max_length_in_tokens is not None:
                result[key] = result[key].filter(lambda example: example['length'] <= max_length_in_tokens)
    
    for key, dataset in result.items():
        print(f"Dataset {key} has {len(dataset)} examples.")
    return result

def tokenize_function(examples, tokenizer, max_len, use_special_tokens=True):
    n_examples = len(examples["prompt"])
    preferred_input_ids = torch.full(
        (n_examples, max_len), tokenizer.pad_token_id, dtype=torch.long
    )
    dispreferred_input_ids = torch.full(
        (n_examples, max_len), tokenizer.pad_token_id, dtype=torch.long
    )
    preferred_attention_masks = torch.zeros((n_examples, max_len), dtype=torch.long)
    dispreferred_attention_masks = torch.zeros((n_examples, max_len), dtype=torch.long)
    for i, (prompt, preferred, dispreferred) in enumerate(
        zip(examples["prompt"], examples["preferred"], examples["dispreferred"])
    ):
        prompt_tokenized = tokenizer.encode(
            prompt, add_special_tokens=use_special_tokens, return_tensors="pt"
        ).view(
            -1
        )  # if special tokens, prompt will have CLS and end with SEP

        preferred_tokenized = tokenizer.encode(
            preferred, add_special_tokens=False, return_tensors="pt"
        ).view(
            -1
        )  # response should not have CLS, it continues the prompt
        prompt_with_preferred = torch.cat((prompt_tokenized, preferred_tokenized))[
            :max_len
        ]
        preferred_mask = torch.cat(
            (
                torch.ones(len(prompt_with_preferred)),
                torch.zeros(max_len - len(prompt_with_preferred)),
            )
        )
        preferred_input_ids[i, : len(prompt_with_preferred)] = prompt_with_preferred
        preferred_attention_masks[i, :] = preferred_mask

        dispreferred_tokenized = tokenizer.encode(
            dispreferred, add_special_tokens=False, return_tensors="pt"
        ).view(
            -1
        )  # same story here
        prompt_with_dispreferred = torch.cat(
            (prompt_tokenized, dispreferred_tokenized)
        )[:max_len]
        dispreferred_mask = torch.cat(
            (
                torch.ones(len(prompt_with_dispreferred)),
                torch.zeros(max_len - len(prompt_with_dispreferred)),
            )
        )
        dispreferred_input_ids[
            i, : len(prompt_with_dispreferred)
        ] = prompt_with_dispreferred
        dispreferred_attention_masks[i, :] = dispreferred_mask

    return {
        "preferred_input_ids": preferred_input_ids,
        "preferred_attention_masks": preferred_attention_masks,
        "dispreferred_input_ids": dispreferred_input_ids,
        "dispreferred_attention_masks": dispreferred_attention_masks,
    }


def build_steamshp_prompt(prompt, a, b, tokenizer, max_len=512):
    prompt_tokens = tokenizer.encode(["POST: " + prompt], return_tensors="pt").view(-1)
    a_tokens = tokenizer.encode(
        ["\n\nRESPONSE A: " + a + "\n\n"], return_tensors="pt"
    ).view(-1)
    b_tokens = tokenizer.encode(
        ["RESPONSE B: " + b + "\n\n"], return_tensors="pt"
    ).view(-1)
    end_tokens = tokenizer.encode(
        ["Which response is better? RESPONSE"], return_tensors="pt"
    ).view(-1)
    all_but_prompt = torch.cat((a_tokens, b_tokens, end_tokens))
    if len(all_but_prompt) > max_len:
        return all_but_prompt[-max_len:]
    elif len(all_but_prompt) + len(prompt_tokens) > max_len:
        tokens_left = max_len - len(all_but_prompt)
        return torch.cat((prompt_tokens[-tokens_left:], all_but_prompt))
    else:
        return torch.cat((prompt_tokens, all_but_prompt))


def tokenize_fn_for_steamshp(examples, tokenizer, max_len=512):
    n_examples = len(examples["prompt"])
    input_ids = torch.full(
        (2 * n_examples, max_len), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_masks = torch.zeros(
        (2 * n_examples, max_len), tokenizer.pad_token_id, dtype=torch.long
    )

    for i, (prompt, preferred, dispreferred) in enumerate(
        zip(examples["prompt"], examples["preferred"], examples["dispreferred"])
    ):
        # create prompts for each ordering of answers, we'll later average over them
        full_text_1 = build_steamshp_prompt(
            prompt, preferred, dispreferred, tokenizer, max_len
        )
        mask_1 = torch.cat(
            (torch.ones(len(full_text_1)), torch.zeros(max_len - len(full_text_1)))
        )
        input_ids[2 * i, : len(full_text_1)] = full_text_1
        attention_masks[2 * i, :] = mask_1

        full_text_2 = build_steamshp_prompt(
            prompt, dispreferred, preferred, tokenizer, max_len
        )
        mask_2 = torch.cat(
            (torch.ones(len(full_text_2)), torch.zeros(max_len - len(full_text_2)))
        )
        input_ids[2 * i + 1, : len(full_text_2)] = full_text_2
        attention_masks[2 * i + 1, :] = mask_2

    return {"input_ids": input_ids, "attention_masks": attention_masks}


def get_train_dataloader(
        datasets,
        add_human_assistant_labels,
        batch_size, 
        tokenizer,
        filter_min_length_in_tokens=None,
        filter_max_length_in_tokens=None,
        seq_len=1024,
):
    if tokenizer is None:
        raise ValueError("Must specify tokenizer.")

    datasets = get_datasets(
        datasets,
        add_human_assistant_labels,
        min_length_in_tokens=filter_min_length_in_tokens, 
        max_length_in_tokens=filter_max_length_in_tokens, 
        tokenizer=tokenizer,
        train=True
    )
    
    if len(datasets) > 1:
        print("Interleaving datasets (this may take a while)...")
        lengths = [len(dataset) for dataset in datasets.values()]
        probabilities = [length / sum(lengths) for length in lengths]
        print([f"{n}: {l}" for n, l in zip(datasets.keys(),lengths)])
        interleaved =  interleave_datasets([d for _, d in datasets.items()], probabilities=probabilities, seed=42)
    else:
        interleaved = datasets[list(datasets.keys())[0]]
    
    print("Tokenizing...")
    tokenized = interleaved.map(
        partial(tokenize_function, tokenizer=tokenizer, max_len=seq_len),
        batched=True,
        batch_size=1000,
        remove_columns=interleaved.column_names,
    )

    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=DefaultDataCollator(),
        num_workers=4,
    )
    
    return dataloader


def to_eval_dataloader(dataset, tokenizer, bsz, max_len, subsample_rate, steamshp=False):
    if subsample_rate is not None and subsample_rate > 1:
        dataset = dataset.select(range(0, len(dataset), subsample_rate))
    if steamshp:
        tokenized = dataset.map(
            partial(tokenize_fn_for_steamshp, tokenizer=tokenizer, max_len=max_len),
            batched=True,
            remove_columns=dataset.column_names,
        )
    else:
        tokenized = dataset.map(
            partial(tokenize_function, tokenizer=tokenizer, max_len=max_len),
            batched=True,
            remove_columns=dataset.column_names,
        )
    return torch.utils.data.DataLoader(
        tokenized,
        batch_size=bsz,
        pin_memory=True,
        collate_fn=DefaultDataCollator(),
        num_workers=4,
    )


def get_eval_dataloaders(tokenizer, bsz, max_len, subsample_rate=10, steamshp=False):
    eval_datasets = get_datasets(datasets="all", tokenizer=tokenizer, max_length_in_tokens=1024, train=False)
    shp, hh, alpaca_gpt4, alpaca_human = (
        eval_datasets["shp"],
        eval_datasets["hh"],
        eval_datasets["alpaca_gpt4"],
        eval_datasets["alpaca_human"],
    )

    return {
        "shp": to_eval_dataloader(shp, tokenizer, bsz, max_len, subsample_rate=subsample_rate, steamshp=steamshp),
        "hh": to_eval_dataloader(hh, tokenizer, bsz, max_len, subsample_rate=subsample_rate, steamshp=steamshp),
        "alpaca_gpt4": to_eval_dataloader(alpaca_gpt4, tokenizer, bsz, max_len, subsample_rate=subsample_rate, steamshp=steamshp),
        "alpaca_human": to_eval_dataloader(alpaca_human, tokenizer, bsz, max_len, subsample_rate=subsample_rate, steamshp=steamshp),
    }

def prepare_data(
    tokenizer_name="microsoft/deberta-v3-base",
    train_datasets=["hh"],
    add_human_assistant_labels=[],
    filter_min_length_in_tokens=None,
    filter_max_length_in_tokens=1024,
    seq_len=1024,
    microbatch_size=16,
    eval_subsample_rate=10,
    data_dir="data",      
):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("=== PREPARING TRAIN DATALOADER ===")
    train_dataloader = get_train_dataloader(
        train_datasets, 
        add_human_assistant_labels,
        microbatch_size, 
        tokenizer,
        filter_min_length_in_tokens,
        filter_max_length_in_tokens,
        seq_len
    )
    torch.save({
        "train_dataloader": train_dataloader,
        "datasets": train_datasets,
        }, os.path.join(save_dir, "train_dataloader.pt"))

    print("=== PREPARING EVAL DATALOADERS ===")
    eval_dataloaders = get_eval_dataloaders(tokenizer, microbatch_size * 4, seq_len, subsample_rate=eval_subsample_rate)
    torch.save(eval_dataloaders, os.path.join(save_dir, "eval_dataloaders.pt"))



if __name__ == "__main__":
    fire.Fire(prepare_data)