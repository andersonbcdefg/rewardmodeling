import os
import time
import tqdm
import uuid
import json
from functools import partial
from datasets import load_dataset
from functional import seq
import json
import openai
from multiprocessing import Pool
openai.api_key = os.environ["OPENAI_API_KEY"]

user = str(uuid.uuid4())

def write_to_file(line, file_path):
    with open(file_path, "a") as f:
        f.write(line)
cb = partial(write_to_file, file_path="synthetic_data/sharegpt_davinci002_responses.jsonl")

def get_completion_text(query):
    # Get completion from text-davinci-002
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=query,
        max_tokens=800,
        temperature=0.25,
        user=user,
        frequency_penalty=0.25
    )

    completion = response.choices[0]['text']
    result = {
        'prompt': query,
        'response': completion.strip()
    }
    return json.dumps(result) + '\n'

if __name__ == '__main__':
    sharegpt_conversations = load_dataset("theblackcat102/sharegpt-english", split="train")['conversations']
    share_gpt_prompts = seq(sharegpt_conversations).map(
        lambda x: x[0]['text']
    ).filter(
        lambda x: len(x) < 2000
    ).to_list()

    pool = Pool(48)
    for prompt in tqdm.tqdm(list(set(share_gpt_prompts))[34000:]):
        prompt = prompt.strip()
        if prompt.endswith("Share Prompt"):
            prompt = prompt[:-len("Share Prompt")]
        pool.apply_async(get_completion_text, args=(prompt,), callback=cb)
        # result = get_completion_text(prompt)
        # cb(result)
        time.sleep(0.2)
    
    pool.close()
    pool.join()
    print("Done!")