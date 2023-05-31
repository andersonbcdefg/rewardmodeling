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

if __name__ == '__main__':

    ai_vs_ai = pd.DataFrame.from_records(read_jsonl("synthetic_data/ranked_responses3.jsonl"))
    human_vs_ai = pd.DataFrame.from_records(read_jsonl("synthetic_data/dolly_ai_vs_human.jsonl"))
    
    # use regex to extract Preferred response: Response A or B, if neither then "Neither"
    ai_vs_ai['preference_clean'] = ai_vs_ai.apply(lambda row: extract_preference(row['preference']), axis=1)
    human_vs_ai['preference_clean'] = human_vs_ai.apply(lambda row: extract_preference(row['preference']), axis=1)

    # filter out responses where neither was clearly preferred
    ai_vs_ai = ai_vs_ai[ai_vs_ai['preference_clean'] != "Neither"]
    human_vs_ai = human_vs_ai[human_vs_ai['preference_clean'] != "Neither"]

    # filter out responses where either response was blank
    ai_vs_ai = ai_vs_ai[ai_vs_ai['response_a'] != ""]
    ai_vs_ai = ai_vs_ai[ai_vs_ai['response_b'] != ""]
    human_vs_ai = human_vs_ai[human_vs_ai['response_a'] != ""]
    human_vs_ai = human_vs_ai[human_vs_ai['response_b'] != ""]

    # dedupe by prompt
    ai_vs_ai = ai_vs_ai.groupby('prompt').first().reset_index()
    human_vs_ai = human_vs_ai.groupby('prompt').first().reset_index()

    # concatenate into one dataset
    combined = pd.concat([ai_vs_ai, human_vs_ai], ignore_index=True).reset_index(drop=True)
    combined.columns = ['prompt', 'response_a', 'response_b', 'explanation', 'preferred']

    # create huggingface dataset and push to hub
    hf = Dataset.from_pandas(combined)
    hf.push_to_hub("andersonbcdefg/dolly_reward_modeling_pairwise")
