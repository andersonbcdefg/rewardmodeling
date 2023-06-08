import json
from datasets import Dataset
import pandas as pd

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


if __name__ == '__main__':

    ranks = pd.DataFrame.from_records(read_jsonl("synthetic_data/red_teaming_eval_gpt4.json"))
    print("Before filtering:", len(ranks))

    # filter out responses where GPT4 disagrees with annotator
    disagreed = ranks[ranks['preference_clean'] != "A"]
    print("Disagreed:", disagreed['prompt'])
    ranks = ranks[ranks['preference_clean'] == "A"]
    print("After filtering:", len(ranks))

    # filter out responses where either response was blank
    ranks = ranks[ranks['response_a'] != ""]
    ranks = ranks[ranks['response_b'] != ""]
    ranks.columns = ['prompt', 'response_a', 'response_b', 'explanation', 'preferred']

    # Deduplicate by prompt
    ranks = ranks.drop_duplicates(subset=['prompt'])

    # create huggingface dataset and push to hub
    hf = Dataset.from_pandas(ranks)
    hf.push_to_hub("andersonbcdefg/redteaming_eval_pairwise")
