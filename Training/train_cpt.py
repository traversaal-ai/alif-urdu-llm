import os
import wandb
import argparse
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments


# Argument Parsing
parser = argparse.ArgumentParser(description="Train PEFT with Wikipedia data.")
parser.add_argument("--model", type=str, required=True, help="Path to the pretrained model.")

# Securely Load API Key
os.environ["HF_TOKEN"] = ""
os.environ["WANDB_API_KEY"] = ""

torch.cuda.set_device(0)

# Pretraining Prompt Template
# pretraining_prompt = """ویکیپیڈیا آرٹیکل
# ### عنوان: {}

# ### مضمون:
# {}"""

pretraining_prompt_ur_wiki = """ویکیپیڈیا آرٹیکل
### عنوان: {}

### مضمون:
{}"""

# pretraining_prompt_ur_wiki = """ویکیپیڈیا آرٹیکل
# ### مضمون:
# {} - {}"""

pretraining_prompt_en_wiki = """### Title: {}

### Article:
{}"""

# pretraining_prompt_cultura = """### مضمون:
# {}"""

pretraining_prompt_cultura = """دلچسپ خبریں
### مضمون:
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


def get_model_peft(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 #default 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",

                          "embed_tokens", "lm_head",], # Add for continual pretraining
        lora_alpha = 32, #default 32
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,   # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model

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

            # Use warmup_ratio and num_train_epochs for longer runs!
            # max_steps = 12,
            warmup_steps = 50,
            # warmup_ratio = 0.1,
            num_train_epochs = 1,

            # Select a 2 to 10x smaller learning rate for the embedding matrices!
            learning_rate = 2e-5,
            embedding_learning_rate = 1e-5,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01, #Suggested 0.01
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
    #Set torch device to get properties global: torch.cuda.set_device(0)
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
    
#     titles = examples["title"]
#     texts = examples["text"]
    
#     #outputs = [pretraining_prompt.format(title, text) + eos_token for title, text in zip(titles, texts)]
#     outputs = []
#     for title, text in zip(titles, texts):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = pretraining_prompt.format(title, text) + eos_token
#         outputs.append(text)
#     return {"text": outputs}

def formatting_prompts_func_ur_wiki(examples, eos_token):

    titles = examples["title"]
    texts  = examples["input_text"]
    outputs = []
    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = pretraining_prompt_ur_wiki.format(title, text) + eos_token
        outputs.append(text)
    return { "text" : outputs, }

def formatting_prompts_func_en_wiki(examples, eos_token):

    titles = examples["title"]
    texts  = examples["input_text"]
    outputs = []
    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = pretraining_prompt_en_wiki.format(title, text) + eos_token
        outputs.append(text)
    return { "text" : outputs, }

def formatting_prompts_func_cultura(examples, eos_token):

    texts = examples["input_text"]

    #outputs = [pretraining_prompt.format(title, text) + eos_token for title, text in zip(titles, texts)]
    outputs = []
    for text in texts:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = pretraining_prompt_cultura.format(text) + eos_token
        outputs.append(text)
    return {"text": outputs}

def load_data(tokenizer):
    
    eos_token = tokenizer.eos_token

    ur_dataset_wiki = load_dataset("wikimedia/wikipedia", "20231101.ur", split="train")
    print(f"Ur Dataset Wiki examples before Train and Split: {ur_dataset_wiki}") # Single Dictionary
    #ur_dataset_wiki = ur_dataset_wiki.train_test_split(train_size=0.999999)["train"] # Only take Train
    print(f"Ur Dataset Wiki examples After Train and Split: {ur_dataset_wiki}") # Single Dictionary
    dataset1 = ur_dataset_wiki.remove_columns([col for col in ur_dataset_wiki.column_names if col not in ["text", "title"]])
    urdu_dataset_wiki = dataset1.rename_column("text", "input_text")
    print(urdu_dataset_wiki)
    print(f"Urdu Dataset Wiki Size in Bytes: {ur_dataset_wiki.dataset_size}") # Single Dictionary
    urdu_dataset_wiki = urdu_dataset_wiki.map(lambda x: formatting_prompts_func_ur_wiki(x, eos_token), batched=True)
    print(urdu_dataset_wiki)

    # en_dataset_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    # print(f"En Dataset Wiki examples before Train and Split: {en_dataset_wiki}") # Single Dictionary
    # en_dataset_wiki = en_dataset_wiki.train_test_split(train_size=0.005)["train"] # Only take Train
    # print(f"En Dataset Wiki examples After Train and Split: {en_dataset_wiki}") # Single Dictionary
    # dataset1 = en_dataset_wiki.remove_columns([col for col in en_dataset_wiki.column_names if col not in ["text", "title"]])
    # english_dataset_wiki = dataset1.rename_column("text", "input_text")
    # print(english_dataset_wiki)
    # print(f"English Dataset Wiki Size in Bytes: {en_dataset_wiki.dataset_size}") # Single Dictionary
    # english_dataset_wiki = english_dataset_wiki.map(lambda x: formatting_prompts_func_en_wiki(x, eos_token), batched=True)
    # print(english_dataset_wiki)

    # ur_dataset_cultura = load_dataset("uonlp/CulturaX", "ur", split="train")
    # print(f"Ur Dataset Cultura examples before Train and Split: {ur_dataset_cultura}") # Single Dictionary
    # ur_dataset_cultura = ur_dataset_cultura.train_test_split(train_size=0.025)["train"] # Only take Train
    # print(f"Ur Dataset Cultura examples After Train and Split: {ur_dataset_cultura}") # Single Dictionary
    # dataset1 = ur_dataset_cultura.remove_columns([col for col in ur_dataset_cultura.column_names if col != "text"])
    # dataset2 = dataset1.map(lambda x: {"title": ""}, num_proc=16)
    # urdu_dataset_cultura = dataset2.rename_column("text", "input_text")
    # print(urdu_dataset_cultura)
    # print(f"Urdu Dataset Cultura Size in Bytes: {ur_dataset_cultura.dataset_size}") # Single Dictionary
    # urdu_dataset_cultura = urdu_dataset_cultura.map(lambda x: formatting_prompts_func_cultura(x, eos_token), batched=True)
    # print(urdu_dataset_cultura)

    # Concatenate the datasets
    combined_dataset = urdu_dataset_wiki #concatenate_datasets([urdu_dataset_wiki, urdu_dataset_cultura])

    # Shuffle the combined dataset
    cont_dataset = combined_dataset.shuffle(seed=42)
    cont_dataset = cont_dataset.shuffle(seed=192)
    cont_dataset = cont_dataset.shuffle(seed=123)
    cont_dataset = cont_dataset.shuffle(seed=487)
    cont_dataset = cont_dataset.shuffle(seed=987)

    print(f"Continued pretraining combined Dataset: {cont_dataset}")

    # Print some examples from the processed dataset
    for i, example in enumerate(cont_dataset[:10]["text"]):  # Adjust the range for more examples
        print(f"Example {i + 1}:\n{example}\n")
    
    return cont_dataset


def cpt_pipeline(model_path: str):
    last_part = os.path.basename(model_path)
    folder_name = last_part + "_ContPretrain"

    """Runs the entire pipeline."""
    model, tokenizer = load_hf_model(model_path)
    print_trainable_parameters(model)

    model = get_model_peft(model)
    print_trainable_parameters(model)
    
    dataset = load_data(tokenizer)

    setup_wandb(folder_name)
    trainer = get_trainer(model, tokenizer, dataset, folder_name)
    
    initial_gpu_memory, max_memory = start_gpu_stat()
    trainer_stats = trainer.train()
    final_gpu_stat(trainer_stats, initial_gpu_memory, max_memory)

    print_trainable_parameters(model)
    
    model.save_pretrained(folder_name) # Local saving
    tokenizer.save_pretrained(folder_name)
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    cpt_pipeline(model_path=args.model)
