import json
import os
import torch
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
        # Check if files are nested inside a checkpoint folder (e.g., checkpoint-120)
        target_path = adapter
        checkpoint_dirs = [os.path.join(adapter, d) for d in os.listdir(adapter) if os.path.isdir(os.path.join(adapter, d)) and d.startswith("checkpoint")]
        if checkpoint_dirs:
            target_path = checkpoint_dirs[0]  # Dynamically use the found checkpoint folder path

        # Initialize the config dict first so it can be updated safely
        adapter_config = {
            "adapter_path": adapter,
            "resolved_path": target_path,
            "lora_config": None,
            "model_config": None
        }

        if load_lora:
            # # # load the adapter using the adapter directory and the adapter name
            
            # # lora_config = PeftConfig.from_pretrained(os.path.join(adapter_directory, adapter))
            # # load the adapter using the absolute adapter path string
            # lora_config = PeftConfig.from_pretrained(adapter)

            # # retrive the lora config as a dictionary
            # lora_config_dict = lora_config.to_dict()
            # adapter_config["lora_config"] = lora_config_dict

            config_json_path = os.path.join(target_path, "adapter_config.json")
            if os.path.exists(config_json_path):
                lora_config = PeftConfig.from_pretrained(target_path)
                adapter_config["lora_config"] = lora_config.to_dict()
            else:
                print(f"adapter.json not found in {target_path}. Skipping LoRA config extraction")

        # # load the model configurations
        # model_config = TrainingArguments.from_pretrained(adapter)

        # adapter_config = {
        #     "adapter_path": adapter,
        #     "model_config": model_config.to_dict()
        # }

        # load model configs from resolved layout paths
        # if os.path.exists(os.path.join(target_path, "training_args.bin")):
        #     model_config = TrainingArguments.from_pretrained(target_path)
        #     adapter_config["model_config"] = model_config.to_dict()
        # else:
        #     print(f"training_args.bin not found in {target_path}. Skipping model config exteaction")

        args_bin_path = os.path.join(target_path, "training_args.bin")
        if os.path.exists(args_bin_path):
            try:
                # load using pytorch
                model_config = torch.load(args_bin_path, map_location="cpu", weights_only=False) # set to false since trained using unsloth which is custom
                # Handle conversion safely if it's already an object or a dictionary
                adapter_config["model_config"] = model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config)
            except Exception as e:
                print(f"Failed to parse training_args.bin in {target_path}: {e}")
        else:
            print(f"training_args.bin not found in {target_path}. Skipping model config exteaction")        

        adapter_configs.append(adapter_config)
    
    return adapter_configs

def main():
    sft_configs = extract_config(SFT_ADAPTERS_DIR, load_lora=True)
    dpo_configs = extract_config(DPO_ADAPTERS_DIR, load_lora=False)

    # print(sft_configs)
    # print("="*40)
    # print(dpo_configs)

    with open("sft_models_configs2.json", "w") as sftFile:
        json.dump(sft_configs, sftFile, indent=4, default=str)
    with open("dpo_models_configs2.json", "w") as dpoFile:
        json.dump(dpo_configs, dpoFile, indent=4, default=str)


if __name__ == "__main__":
    main()
