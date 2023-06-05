import re
from functools import partial
from itertools import chain
import torch
# import matplotlib.pyplot as plt
from datasets import load_dataset, interleave_datasets
from transformers import DefaultDataCollator


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


def get_train_datasets(
        datasets=["hh"], 
        add_human_assistant_labels=[],
        min_length_in_tokens=None, 
        max_length_in_tokens=None, 
        tokenizer=None
):
    if tokenizer is None:
        if min_length_in_tokens is not None or max_length_in_tokens is not None:
            raise ValueError("Cannot specify min_length_in_tokens or max_length_in_tokens without specifying tokenizer.")
    
    if datasets == "all":
        datasets = ["hh", "shp", "webgpt", "summarize", "synth_gptj", "synth_dolly", "synth_redteam", "synth_gpteacher"]

    result = {}
    # process all datasets to have the same 3 columns: prompt, preferred, dispreferred
    if "hh" in datasets:
        hh_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        hh_dataset = hh_dataset.map(
            process_anthropic, remove_columns=hh_dataset.column_names
        )
        result["hh"] = hh_dataset

    if "shp" in datasets:
        shp_dataset = load_dataset("stanfordnlp/SHP", split="train")
        shp_dataset = shp_dataset.filter(
            lambda example: example["score_ratio"] >= 2.0 and example["score_ratio"] <= 8.0
        ).map(process_shp, remove_columns=shp_dataset.column_names)
        result["shp"] = shp_dataset

    if "webgpt" in datasets:
        webgpt_dataset = load_dataset("openai/webgpt_comparisons", split="train", download_mode="force_redownload")
        webgpt_dataset = webgpt_dataset.map(
            process_webgpt, remove_columns=webgpt_dataset.column_names
        ).filter(
            lambda example: example["preferred"] != "" and example["dispreferred"] != ""
        )
        result["webgpt"] = webgpt_dataset

    if "summarize" in datasets:
        pass

    if "synth_gptj" in datasets:
        synth_gptj_dataset = load_dataset('Dahoas/synthetic-instruct-gptj-pairwise', split='train')
        synth_gptj_dataset = synth_gptj_dataset.map(
            lambda example: {
                "prompt": example["prompt"],
                "preferred": example["chosen"],
                "dispreferred": example["rejected"],
            },
            remove_columns=synth_gptj_dataset.column_names
        )
        result["synth_gptj"] = synth_gptj_dataset

    if "synth_dolly" in datasets:
        synth_dolly_dataset = load_dataset("andersonbcdefg/dolly_reward_modeling_pairwise", split="train")
        synth_dolly_dataset = synth_dolly_dataset.map(process_synthetic, remove_columns=synth_dolly_dataset.column_names)
        result["synth_dolly"] = synth_dolly_dataset

    if "synth_redteam" in datasets:
        synth_redteam_dataset = load_dataset("andersonbcdefg/red_teaming_reward_modeling_pairwise_no_as_an_ai", split="train")
        synth_redteam_dataset = synth_redteam_dataset.map(process_synthetic, remove_columns=synth_redteam_dataset.column_names)
        result["synth_redteam"] = synth_redteam_dataset

    if "synth_gpteacher" in datasets:
        pass

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


def get_eval_datasets(datasets=["hh"], min_length_in_tokens=None, max_length_in_tokens=None, tokenizer=None):
    if tokenizer is None:
        if min_length_in_tokens is not None or max_length_in_tokens is not None:
            raise ValueError("Cannot specify min_length_in_tokens or max_length_in_tokens when specifying tokenizer.")
        
    if datasets == "all":
        datasets = ["hh", "shp", "alpaca_gpt4", "alpaca_human"]

    result = {}
    
    if "hh" in datasets:
        hh_dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        hh_dataset = hh_dataset.map(
            process_anthropic, remove_columns=hh_dataset.column_names
        )
        result["hh"] = hh_dataset

    if "shp" in datasets:
        shp_dataset = load_dataset("stanfordnlp/SHP", split="validation")
        shp_dataset = shp_dataset.map(process_shp, remove_columns=shp_dataset.column_names)
        result["shp"] = shp_dataset
    
    if "alpaca_gpt4" in datasets:
        alpaca_gpt4_dataset = load_dataset(
            "tatsu-lab/alpaca_farm", "alpaca_gpt4_preference", split="preference"
        )
        alpaca_gpt4_dataset = alpaca_gpt4_dataset.map(
            process_alpaca, remove_columns=alpaca_gpt4_dataset.column_names
        )
        result["alpaca_gpt4"] = alpaca_gpt4_dataset

    if "alpaca_human" in datasets:
        alpaca_human_dataset = load_dataset(
            "tatsu-lab/alpaca_farm", "alpaca_human_preference", split="preference"
        )
        alpaca_human_dataset = alpaca_human_dataset.map(
            process_alpaca, remove_columns=alpaca_human_dataset.column_names
        )
        result["alpaca_human"] = alpaca_human_dataset

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
        batch_size, 
        tokenizer,
        filter_min_length_in_tokens=None,
        filter_max_length_in_tokens=None,
        seq_len=1024
):
    if tokenizer is None:
        raise ValueError("Must specify tokenizer.")

    datasets = get_train_datasets(datasets, filter_min_length_in_tokens, filter_max_length_in_tokens, tokenizer)
    lengths = [len(dataset) for dataset in datasets.values()]
    probabilities = [length / sum(lengths) for length in lengths]
    print("Interleaving datasets (this may take a while)...")
    interleaved =  interleave_datasets([d for _, d in datasets.items()], probabilities=probabilities, seed=42)
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
        pin_memory=True,
        collate_fn=DefaultDataCollator(),
        num_workers=4,
    )
    
    return dataloader


def to_eval_dataloader(dataset, tokenizer, bsz, max_len, steamshp=False):
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


def get_eval_dataloaders(tokenizer, bsz, max_len, steamshp=False):
    eval_datasets = get_eval_datasets(datasets="all", tokenizer=tokenizer, max_length_in_tokens=1024)
    shp, hh, alpaca_gpt4, alpaca_human = (
        eval_datasets["shp"],
        eval_datasets["hh"],
        eval_datasets["alpaca_gpt4"],
        eval_datasets["alpaca_human"],
    )
    shp_loader, hh_loader, alpaca_gpt4_loader, alpaca_human_loader = map(
        lambda x: to_eval_dataloader(x, tokenizer, bsz, max_len, steamshp=steamshp),
        [shp, hh, alpaca_gpt4, alpaca_human],
    )
    return {
        "shp": shp_loader,
        "hh": hh_loader,
        "alpaca_gpt4": alpaca_gpt4_loader,
        "alpaca_human": alpaca_human_loader,
    }
