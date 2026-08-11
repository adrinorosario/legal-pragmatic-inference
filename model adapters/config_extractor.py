import json
import os
from peft import PeftConfig
from transformers import TrainingArguments
from pathlib import Path

SFT_ADAPTERS_DIR = "./sft_adapters"
DPO_ADAPTERS_DIR = "./dpo_adapters"

def extract_config(adapter_directory: str, load_lora: bool = False) -> list:
    dir_path = Path(adapter_directory)

    adapters = [os.path.abspath(os.path.join(adapter_directory, entry.name)) for entry in dir_path.iterdir() if entry.is_dir()]


    adapter_configs = []

    for adapter in adapters:

        if load_lora:
            # # load the adapter using the adapter directory and the adapter name
            
            # lora_config = PeftConfig.from_pretrained(os.path.join(adapter_directory, adapter))
            # load the adapter using the absolute adapter path string
            lora_config = PeftConfig.from_pretrained(adapter)

            # retrive the lora config as a dictionary
            lora_config_dict = lora_config.to_dict()
            adapter_config["lora_config"] = lora_config_dict

        # load the model configurations
        model_config = TrainingArguments.from_pretrained(adapter)

        adapter_config = {
            "adapter_path": adapter,
            "model_config": model_config.to_dict()
        }

        adapter_configs.append(adapter_config)
    
    return adapter_configs

def main():
    sft_configs = extract_config(SFT_ADAPTERS_DIR, load_lora=True)
    dpo_configs = extract_config(DPO_ADAPTERS_DIR, load_lora=False)

    print(sft_configs)
    print("="*40)
    print(dpo_configs)


if __name__ == "__main__":
    main()
