from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk
import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# read in the locally saved hf dataset
sft_dataset = load_from_disk("../datasets/sft_dataset_local")

max_length = 2048

# # Standard loading architecture that correctly handles fallback parsing on Mac/CPU
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"
# )

# Pass an explicit dtype parameter to prevent the MLX validation lookup failure
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-12B",
    load_in_4bit=True,
    dtype="float16" # Explicitly locks the datatype variable mapping
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",  # Bias = "none" is currently optimized
    use_gradient_checkpointing=True,
    random_state=3407,
)

training_args = SFTConfig(output_dir="./output", max_length=max_length)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
)

trainer.train()