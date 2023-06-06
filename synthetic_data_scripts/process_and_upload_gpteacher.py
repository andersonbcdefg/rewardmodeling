import re
import json
from datasets import Dataset
import pandas as pd

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


if __name__ == '__main__':

    gpteacher_ranks = pd.DataFrame.from_records(read_jsonl("synthetic_data/ranked_gpteacher.jsonl"))
    
    # filter out responses where neither was clearly preferred
    gpteacher_ranks = gpteacher_ranks[gpteacher_ranks['preference_clean'] != "Neither"]

    # filter out responses where either response was blank
    gpteacher_ranks = gpteacher_ranks[gpteacher_ranks['response_a'] != ""]
    gpteacher_ranks.columns = ['prompt', 'response_a', 'response_b', 'explanation', 'preferred']

    # create huggingface dataset and push to hub
    hf = Dataset.from_pandas(gpteacher_ranks)
    hf.push_to_hub("andersonbcdefg/gpteacher_reward_modeling_pairwise")
