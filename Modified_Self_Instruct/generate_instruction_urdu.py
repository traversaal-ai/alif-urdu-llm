
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import openai
from openai import OpenAI
from openai import AzureOpenAI
from LughaatNLP import LughaatNLP

openai.api_key = ''
os.environ["OPENAI_API_KEY"] = ''
AZURE_OPENAI_ENDPOINT = ''
AZURE_OPENAI_KEY = '' 
os.environ['AZURE_OPENAI_ENDPOINT'] = AZURE_OPENAI_ENDPOINT
os.environ['AZURE_OPENAI_KEY'] = AZURE_OPENAI_KEY

#client = OpenAI() #change model to gpt-4o
client = AzureOpenAI(  #change model to gpt-4o-large
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_KEY"),
  api_version=""
)
urdu_text_processing = LughaatNLP()
#tokenizer=lambda x: x.split()

def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt_urdu.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"Instruction: {instruction}\n"
        prompt += f"Input:\n{input}\n"
        prompt += f"Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"Instruction:"
    return prompt


def post_process_gpt4_response(num_prompt_instructions, response):
    if response is None:
        return []
    if response.choices[0].message.content is None:
        return []
    
    raw_instructions = response.choices[0].message.content
    
    if not raw_instructions.startswith("Instruction"):
        raw_instructions = f"Instruction: " + response.choices[0].message.content#response["text"]
    
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response.choices[0].finish_reason == "length":
            continue
        idx += num_prompt_instructions + 1
        inst = inst.strip()
        splitted_data = re.split(f"(Instruction|Input|Output):", inst)
        
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        if len(output.split()) < 1:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
    "تصویر",      # image
    "تصاویر",     # images
    "گراف",       # graph
    "گرافز",      # graphs
    "فائل",       # file
    "فائلز",      # files
    "نقشہ",       # map
    "نقشے",       # maps
    "بنائیں",     # draw
    "ڈرائنگ",     # drawing
    "پلاٹ",       # plot
    "جائیں",      # go to
    "ویڈیو",      # video
    "آڈیو",       # audio
    "موسیقی",     # music
    "فلو چارٹ",   # flowchart
    "خاکہ",       # diagram
    "ڈایاگرام"     # diagram
]

        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst is None:
            continue
        if inst.startswith("ایک پروگرام لکھیں"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not is_ascii_or_urdu(inst[0]): #if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def is_ascii_or_urdu(char):
    # Check if character is ASCII or in Urdu Unicode range
    return char.isascii() or '\u0600' <= char <= '\u06FF'

# =============================================================================
# def find_word_in_string(w, s):
#     return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)
# =============================================================================
def find_word_in_string(w, s):
    # Adjust pattern to account for Urdu word boundaries
    pattern = r"(^|\s){0}($|\s)".format(re.escape(w))
    return re.compile(pattern, flags=re.IGNORECASE).search(s)


# Function to process a batch of prompts
def get_batch_responses(prompts):
    responses = []
    for prompt in prompts:
        # Send request to OpenAI API
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-large",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                #max_tokens=256,
                top_p=1,
                n=1,
                stop=["\n20", "20.", "20."],
                logit_bias={"50256": -100}  # Prevent <|endoftext|> token generation
                
            )
            responses.append(completion) #completion.choices[0].message.content
            time.sleep(1)  # Optional delay to avoid rate limits
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
            responses.append(None)
    return responses

def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks_urdu.jsonl",
    num_instructions_to_generate=60410,
    num_prompt_instructions=4,
    request_batch_size=4,
    num_cpus=24,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r", encoding="utf-8")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen_urdu.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen_urdu.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    #all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    all_instruction_tokens = [urdu_text_processing.urdu_tokenize(urdu_text_processing.lemmatize_sentence((inst))) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # Sample 4 from seed data and 2 from machine-generated data
            seed_sample = random.sample(seed_instruction_data, 4)
            machine_sample = random.sample(machine_instruction_data, 2)
    
            # Combine the two
            prompt_instructions = seed_sample + machine_sample
            random.shuffle(prompt_instructions)  # Optional: shuffle to mix them

            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
        
        
        request_start = time.time()
        results = get_batch_responses(batch_inputs)
        output_result = results[0].choices[0].message.content
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt4_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            #new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            new_instruction_tokens = urdu_text_processing.urdu_tokenize(urdu_text_processing.lemmatize_sentence((instruction_data_entry["instruction"])))
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            instruction_data_entry["category"] = "test_ethics"
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen_urdu.json"))



def main():
    # Default arguments for debugging
    generate_instruction_following_data()


if __name__ == "__main__":
    main()