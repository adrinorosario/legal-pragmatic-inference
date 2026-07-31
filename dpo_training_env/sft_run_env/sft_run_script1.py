from datasets import load_dataset, load_from_disk, Dataset
import torch
from unsloth import FastLanguageModel,is_bfloat16_supported, get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import accelerate
from accelerate import PartialState

import wandb
import os
import subprocess

import json
import time
import pandas as pd

# login to hf to read the gated models
from huggingface_hub import login
hf_token = os.environ.get("HF_READ_TOKEN")
login(token=hf_token)

def get_available_gpu_ids() -> list[int]:
    """Detect visible CUDA devices without touching the current process's CUDA context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpu_ids = [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip() != ""]
        if not gpu_ids:
            raise ValueError("nvidia-smi returned no GPUs")
        return gpu_ids
    except Exception as e:
        print(f"nvidia-smi detection failed ({e}), falling back to torch.cuda.device_count()")
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]

AVAILABLE_GPUS = get_available_gpu_ids()
print(f"Detected GPUs: {AVAILABLE_GPUS}")

# log in to wandb and set up metric logging for sft train run
def wandb_login():
    # 1. Fetch your secret API key from Kaggle environment
    try:
        wandb_key = os.environ.get("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_key
        # This prevents W&B from prompting for interactive login input
        os.environ["WANDB_SILENT"] = "true"
    except Exception as e:
        print(f"Make sure you added WANDB_API_KEY to .env: {e}")

# used to setup the wandb run logging
def setup_wandb_run_logging(project_name, run_name):
    wandb.init(project=project_name, name=run_name)

# load the local dataset or on the online env
def load_sft_dataset():
    sft_dataset_json_path = "../datasets/sft_dataset_local.json"
    sft_dataset = load_dataset("json", data_files=sft_dataset_json_path)
    return sft_dataset

# 1. Initialize multi-GPU environment placement tracker
device_string = f"cuda:{PartialState().local_process_index}"

def load_model_tokenizer(model_name: str, 
                         maximum_sequence_length: int, 
                         load_in_4_bit: bool,
                        formatting_chat_template_name: str) -> tuple:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=maximum_sequence_length,
        load_in_4bit=load_in_4_bit,# for efficient 4b it loading
        device_map={"": device_string} # Force it onto the specific process GPU
    )

    # Setup PEFT/LoRA modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # set up the chat template for prompt-response
    tokenizer = get_chat_template(
        tokenizer, 
        chat_template=formatting_chat_template_name # configures the template for llama3
    )

    return (model, tokenizer)

# function to format the prompts structure according to the chat template format
def formatting_prompts_func(dataset, tokenizer):
    prompt = dataset["prompt"]
    response = dataset["response"]
    texts = []

    for prompt, response in zip(prompt, response):
        # convert pairs into huggingface conversation turns
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        # apply tokenizer template to produce a single continuous string
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    return {"text": texts}

print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# standard named function (Pickle can serialize this)
def map_text_column(example):
    return example["text"]

def save_training_metrics(trainer):
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv("./output.csv", index= False)

def SFT_TRAINING_SETUP():
    # load the dataset
    sft_dataset = load_sft_dataset()

    # load the model and setup tokenizer
    model, tokenizer = load_model_tokenizer(
        model_name="unsloth/Ministral-3-14B-Reasoning-2512-bnb-4bit",
        maximum_sequence_length=4096,
        load_in_4_bit=True,
        formatting_chat_template_name="mistral"
    )

    # apply format mapping to the sft dataset
    sft_dataset = sft_dataset.map(lambda batch: formatting_prompts_func(batch, tokenizer), batched=True)

    # set the training arguments
    training_args = SFTConfig(
        output_dir="/kaggle/working/",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        warmup_steps=5,
        max_steps=120,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        # optim="paged_adamw_8bit",          # Paged memory allocation prevents peak OOMs
        logging_steps=1,
        average_tokens_across_devices=False,
        max_length=5120,
        ddp_find_unused_parameters=False, # Required optimization flag for DDP training loops
        report_to="wandb", # Silences duplicate multi-process tracking warnings
        eval_strategy="no",
        eval_steps=10,
        per_device_eval_batch_size=4,
    )

    return sft_dataset, model, tokenizer, training_args

def SFT_TRAIN(dataset, model, tokenizer, formatting_function, training_args, run_name):

    setup_wandb_run_logging(
        project_name="kaggle-sft-research",
        run_name=run_name
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        formatting_func=formatting_function,
        args=training_args
    )

    trainer.train()
    wandb.finish()

    return trainer

if __name__ == "__main__":
    sft_dataset, model, tokenizer, training_args = SFT_TRAINING_SETUP()
    trainer = SFT_TRAIN(
        dataset=sft_dataset,
        model=model,
        tokenizer=tokenizer,
        formatting_function=map_text_column,
        training_args=training_args,
        run_name="mistralai/Ministral-3-14B-Reasoning"
    )
    save_training_metrics(trainer)