import re
import json
from datasets import Dataset
import pandas as pd

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_to_file(result):
    with open("dolly_ai_vs_human.jsonl", "a") as f:
        f.write(result)

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

    red_teaming_ranks = pd.DataFrame.from_records(read_jsonl("synthetic_data/ranked_redteam.jsonl"))
    
    # use regex to extract Preferred response: Response A or B, if neither then "Neither"
    red_teaming_ranks['preference_clean'] = red_teaming_ranks.apply(lambda row: extract_preference(row['preference']), axis=1)

    # filter out responses where neither was clearly preferred
    red_teaming_ranks = red_teaming_ranks[red_teaming_ranks['preference_clean'] != "Neither"]

    # filter out responses where either response was blank
    red_teaming_ranks = red_teaming_ranks[red_teaming_ranks['response_a'] != ""]
    red_teaming_ranks.columns = ['prompt', 'response_a', 'response_b', 'explanation', 'preferred']

    # create huggingface dataset and push to hub
    hf = Dataset.from_pandas(red_teaming_ranks)
    hf.push_to_hub("andersonbcdefg/red_teaming_reward_modeling_pairwise")

    # alternate version where mentions of "as an AI language model" are minimized
    red_teaming_ranks['response_a'] = red_teaming_ranks.apply(lambda row: remove_as_an_ai(row['response_a']), axis=1)
    red_teaming_ranks['response_b'] = red_teaming_ranks.apply(lambda row: remove_as_an_ai(row['response_b']), axis=1)
    hf = Dataset.from_pandas(red_teaming_ranks)
    hf.push_to_hub("andersonbcdefg/red_teaming_reward_modeling_pairwise_no_as_an_ai")
