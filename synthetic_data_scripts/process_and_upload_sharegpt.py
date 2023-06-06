import re
import json
from datasets import Dataset
import pandas as pd

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def extract_preference(preference):
    groups = re.findall(r"Preferred response: Response (A|B)", preference)
    if len(groups) > 0:
        return groups[0]
    else:
        return "Neither"

def remove_as_an_ai(response):
    response = re.sub(r"as an AI language model, I", "I", response)
    response = re.sub(r"As an AI language model, I", "I", response)
    return response


if __name__ == '__main__':

    sharegpt_ranks = pd.DataFrame.from_records(read_jsonl("synthetic_data/ranked_sharegpt.jsonl"))
    
    # filter out responses where neither was clearly preferred
    sharegpt_ranks = sharegpt_ranks[sharegpt_ranks['preference_clean'] != "Neither"]

    # filter out responses where either response was blank
    sharegpt_ranks = sharegpt_ranks[sharegpt_ranks['response_a'] != ""]
    sharegpt_ranks.columns = ['prompt', 'response_a', 'response_b', 'explanation', 'preferred']

    # Deduplicate by prompt
    sharegpt_ranks = sharegpt_ranks.drop_duplicates(subset=['prompt'])

    # create huggingface dataset and push to hub
    hf = Dataset.from_pandas(sharegpt_ranks)
    hf.push_to_hub("andersonbcdefg/sharegpt_reward_modeling_pairwise")

    # alternate version where mentions of "as an AI language model" are minimized
    sharegpt_ranks['response_a'] = sharegpt_ranks.apply(lambda row: remove_as_an_ai(row['response_a']), axis=1)
    sharegpt_ranks['response_b'] = sharegpt_ranks.apply(lambda row: remove_as_an_ai(row['response_b']), axis=1)
    hf = Dataset.from_pandas(sharegpt_ranks)
    hf.push_to_hub("andersonbcdefg/sharegpt_reward_modeling_pairwise_no_as_an_ai")
