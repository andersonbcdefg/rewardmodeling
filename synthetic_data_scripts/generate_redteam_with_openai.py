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
template = """\
# The following is a Q&A between a user and a helpful, knowledgeable assistant. The assistant always explains his answers (rather than simply saying yes or no) and responds with a paragraph or two in order to answer the user's questions thoroughly. The assistant is never evasive, and answers honestly and candidly, even if the question is difficult or controversial. When the assistant isn't sure of the answer, or doesn't know exactly what the user means, he provides his best guess at what the user intended.

User: {query}

Assistant:"""

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_to_file(line, file_path):
    with open(file_path, "a") as f:
        f.write(line)
cb = partial(write_to_file, file_path="synthetic_data/redteaming_davinci_responses.jsonl")

def get_completion_chat(query):
    # Get completion from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": query}
        ],
        user=user
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
        model="text-curie-001",
        prompt=template.format(query=query),
        max_tokens=500,
        temperature=0.25,
        user=user,
        frequency_penalty=0.5,
        stop=["User:", "Assistant:"]
    )

    completion = response.choices[0]['text']
    result = {
        'prompt': query,
        'response': completion.strip()
    }
    return json.dumps(result) + '\n'

if __name__ == '__main__':

    red_teaming_prompts = seq(json.load(open("/Users/ben/Downloads/red_team_attempts.jsonl"))).map(
        lambda attempt: attempt['transcript']
    ).map(
        lambda transcript: transcript.split("Assistant:")[0].replace("Human:", "").strip()
    ).to_list()

    pool = Pool(64)

    
    for prompt in tqdm.tqdm(list(set(red_teaming_prompts))[8000:]):
        pool.apply_async(get_completion_chat, args=(prompt,), callback=cb)
        # result = get_completion_text(prompt)
        # cb(result)
        time.sleep(0.1)
    
    pool.close()
    pool.join()
    print("Done!")