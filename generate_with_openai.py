import os
import time
import tqdm
from functools import partial
from datasets import load_dataset
import json
import openai
from multiprocessing import Pool
openai.api_key = os.environ["OPENAI_API_KEY"]

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_to_file(line, file_path):
    with open(file_path, "a") as f:
        f.write(line)
cb = partial(write_to_file, file_path="gpteacher_responses.jsonl")

def get_completion_chat(query):
    # Get completion from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": query}
            ]
    )

    completion = response.choices[0]['message']['content']
    result = {
        'prompt': query,
        'response': completion
    }
    return json.dumps(result) + '\n'

def get_completion_text(query):
    # Get completion from text-davinci-003
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        max_tokens=500,
        temperature=0.25
    )

    completion = response.choices[0]['text']
    result = {
        'prompt': query,
        'response': completion.strip()
    }
    return json.dumps(result) + '\n'

if __name__ == '__main__':

    gpteacher_prompts = load_dataset("teknium/GPTeacher-General-Instruct", split="train").filter(
        lambda example: example['input'] == ""
    )['instruction']

    pool = Pool(16)

    
    for prompt in tqdm.tqdm(gpteacher_prompts):
        pool.apply_async(get_completion_text, args=(prompt,), callback=cb)
        # result = get_completion_text(prompt)
        # cb(result)
    
    pool.close()
    pool.join()
    print("Done!")