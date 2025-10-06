import os
import wandb
import argparse
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import Dataset, concatenate_datasets

# Argument Parsing
parser = argparse.ArgumentParser(description="Finetune with data distillation")
parser.add_argument("--model", type=str, required=True, help="Path to the continued pretrained model.")

# Securely Load API Key
os.environ["HF_TOKEN"] = ""
os.environ["WANDB_API_KEY"] = ""

torch.cuda.set_device(0)

# Finetune Prompt Template
prompt_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

prompt_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

prompt_orca_header = """{}

### Instruction:
{}

### Response:
{}"""

prompt_orca_no_header = """### Instruction:
{}

### Response:
{}"""

# Hyperparameters
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_hf_model(model_path: str):
    """Loads a Hugging Face model and tokenizer."""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=os.getenv("HF_TOKEN"),
        )
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    return model, tokenizer

def setup_wandb(_folder_name):
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]=_folder_name
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="false"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"

def get_trainer(model, tokenizer, dataset, folder):
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 16,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 8,

            # Use num_train_epochs and warmup_ratio for longer runs!
            # max_steps = 120,
            warmup_steps = 50,
            # warmup_ratio = 0.1,
            num_train_epochs = 2,

            # Select a 2 to 10x smaller learning rate for the embedding matrices!
            learning_rate = 5e-5,
            embedding_learning_rate = 1e-5,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.00,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = folder,
            save_total_limit = 2,     # Retain the 3 most recent checkpoints
            report_to = "wandb", # Use this for WandB etc
        ),
    )
    return trainer


def start_gpu_stat():
    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    initial_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    return initial_gpu_memory, max_memory

def final_gpu_stat(_trainer_stats, _initial_gpu_memory, _max_memory):
    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_diff = round(used_memory - _initial_gpu_memory, 3)
    used_percentage = round(used_memory         /_max_memory*100, 3)
    diff_percentage = round(used_memory_for_diff/_max_memory*100, 3)
    print(f"{_trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(_trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

    print(f"Max memory = {_max_memory} GB.")
    print(f"{_initial_gpu_memory} GB of INITIAL memory reserved.")
    print(f"Peak reserved FINAL memory = {used_memory} GB.")
    print(f"Peak reserved memory DIFFERENCE = {used_memory_for_diff} GB.")
    print(f"Peak reserved memory % of FINAL memory = {used_percentage} %.")
    print(f"Peak reserved memory % of DIFFERENCE memory = {diff_percentage} %.")


# def formatting_prompts_func(examples, eos_token):
#     instructions = examples["urdu_instruction"]
#     inputs       = examples["urdu_input"]
#     outputs      = examples["urdu_output"]
#     texts = []
#     for instruction, inp, output in zip(instructions, inputs, outputs):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = finetune_prompt.format(instruction, inp, output) + eos_token
#         texts.append(text)
#     return { "text" : texts, }

def formatting_prompts_func_ur(examples, eos_token):
    instructions = examples["instruction"]
    inputs = examples.get("input", [None] * len(instructions))  # Default to a list of None if missing
    outputs = examples["output"]

    # Ensure all columns have the same length
    if not (len(instructions) == len(inputs) == len(outputs)):
        raise ValueError("Mismatch in column lengths: instructions, inputs, and outputs must be the same length.")

    formatted_prompts = []

    for instruction, input_value, output in zip(instructions, inputs, outputs):
        if input_value:  # Check if input_value is not None and not empty
            formatted_prompt = prompt_with_input.format(instruction, input_value, output)
        else:
            formatted_prompt = prompt_without_input.format(instruction, output)

        # Append the EOS token to ensure proper termination
        formatted_prompts.append(formatted_prompt + eos_token)

    return {"text": formatted_prompts}

def formatting_prompts_func_orca(examples, eos_token):
    headers = examples["instruction"]
    instructions = examples["input"]
    outputs = examples["output"]

    formatted_prompts = []

    for header, instruction, output in zip(headers, instructions, outputs):
        if header:  # Check if header is not None and not empty
            formatted_prompt = prompt_orca_header.format(header, instruction, output)
        else:
            formatted_prompt = prompt_orca_no_header.format(instruction, output)

        # Append the EOS token to ensure proper termination
        formatted_prompts.append(formatted_prompt + eos_token)

    return {"text": formatted_prompts}

def cast_input_to_string(example):
    example["input"] = str(example["input"]) if example["input"] is not None else ""
    return example

# Define the function to append the eos_token
def add_eos_token(example, eos_token):
    example["text"] = example["text"] + eos_token
    return example

def load_data(tokenizer):

    eos_token = tokenizer.eos_token

    urdu_alpaca = load_dataset('csv', data_files='ftdata/urdu-alpaca-data-final.csv')
    urdu_alpaca = urdu_alpaca['train']
    print(f"urdu alpaca: {urdu_alpaca}")
    urdu_alpaca = urdu_alpaca.map(lambda x: formatting_prompts_func_ur(x, eos_token), batched=True)
    urdu_alpaca = urdu_alpaca.map(cast_input_to_string)
    print(f"urdu alpaca: {urdu_alpaca}")

    # james_wyang = load_dataset('csv', data_files='ftdata/test_james_Ali_Final.csv')
    # james_wyang = james_wyang['train']
    # print(f"james wyang: {james_wyang}")
    # james_wyang = james_wyang.map(lambda x: formatting_prompts_func_ur(x, eos_token), batched=True)
    # james_wyang = james_wyang.map(cast_input_to_string)
    # print(f"james wyang: {james_wyang}")

    uls_wsd = load_dataset('csv', data_files='ftdata/uls-wsd-merged-finetune-train-data.csv')
    uls_wsd = uls_wsd['train']
    print(f"uls_wsd: {uls_wsd}")
    uls_wsd = uls_wsd.map(lambda x: formatting_prompts_func_ur(x, eos_token), batched=True)
    uls_wsd = uls_wsd.map(cast_input_to_string)
    print(f"uls_wsd: {uls_wsd}")

    english_alpaca_data = load_dataset("tatsu-lab/alpaca",split="train")
    print(f"english alpaca: {english_alpaca_data}")
    english_alpaca_with_eos = english_alpaca_data.map(lambda example: add_eos_token(example, eos_token))
    english_alpaca = english_alpaca_with_eos.train_test_split(train_size=0.2)["train"]
    english_alpaca = english_alpaca.map(cast_input_to_string)
    print(f"english alpaca: {english_alpaca}")

    open_orca = load_dataset('csv', data_files='ftdata/open-orca-10k-sample.csv')
    open_orca = open_orca['train']
    print(f"open orca: {open_orca}")
    open_orca = open_orca.map(lambda x: formatting_prompts_func_orca(x, eos_token), batched=True)
    open_orca = open_orca.map(cast_input_to_string)
    print(f"open orca: {open_orca}")

    new_self_instruct = load_dataset('csv', data_files='ftdata/new_self_instruct.csv')
    new_self_instruct = new_self_instruct['train']
    print(f"new self instruct: {new_self_instruct}")
    new_self_instruct = new_self_instruct.map(lambda x: formatting_prompts_func_ur(x, eos_token), batched=True)
    new_self_instruct = new_self_instruct.map(cast_input_to_string)
    print(f"new self instruct: {new_self_instruct}")

    # open_orca, new_self_instruct, urdu_alpaca, uls_wsd, english_alpaca
    combined_dataset = concatenate_datasets([urdu_alpaca, new_self_instruct, uls_wsd, english_alpaca, open_orca])

    # Shuffle the combined dataset
    fine_dataset = combined_dataset.shuffle(seed=42)
    fine_dataset = fine_dataset.shuffle(seed=122)
    fine_dataset = fine_dataset.shuffle(seed=372)
    fine_dataset = fine_dataset.shuffle(seed=892)

    print(f"Fine tuned dataset: {fine_dataset}")
    
    # Print some examples from the processed dataset
    for i, example in enumerate(fine_dataset[:50]["text"]):  # Adjust the range for more examples
        print(f"Example {i + 1}:\n{example}\n")
    
    return fine_dataset


def ft_pipeline(model_path: str):
    # Extract the last part of the path
    last_part = os.path.basename(model_path)
    extracted_name = last_part.split("_")[0]
    folder_name = extracted_name + "_CPFinetune"

    """Runs the entire pipeline."""
    model, tokenizer = load_hf_model(model_path)

    adapter_name = model.active_adapter  # Get the current adapter name
    print(f"Adapter name: {adapter_name}")
    model.set_adapter(adapter_name)  
    print_trainable_parameters(model)

    dataset = load_data(tokenizer)

    setup_wandb(folder_name)
    trainer = get_trainer(model, tokenizer, dataset, folder_name)

    initial_gpu_memory, max_memory = start_gpu_stat()
    trainer_stats = trainer.train()
    final_gpu_stat(trainer_stats, initial_gpu_memory, max_memory)
    
    model.save_pretrained(folder_name) # Local saving
    tokenizer.save_pretrained(folder_name)
    # # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    ft_pipeline(model_path=args.model)
