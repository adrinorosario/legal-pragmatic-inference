import json
import os
from datasets import Dataset

DPO_FORMAT_PAIR = []
SFT_FORMAT_PAIR = []

PROMPT_TEMPLATE = (
    "Given the following contractual clause, provide judicial reasoning of "
    "the kind a court would apply when interpreting clauses that raise "
    "similar legal questions:\n\n{prompt_anchor}"
)

def load_jsonl_file(file_path):
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip() # strip off all trailing and leading spaces
                if line: # check if line is not empty
                    # full data
                    full_data = json.loads(line)

                    # extract required keys
                    prompt_anchor = full_data["prompt_anchor"]
                    chosen_response = full_data["chosen"]
                    rejected_response = full_data["rejected"]

                    # formatted prompt anchor
                    formatted_prompt_anchor = PROMPT_TEMPLATE.format(
                        prompt_anchor=prompt_anchor
                    )

                    # add to sft format pair
                    sft_pair = {
                        "prompt": formatted_prompt_anchor,
                        "response": chosen_response
                    }
                    SFT_FORMAT_PAIR.append(sft_pair)

                    # add to dpo format pair
                    dpo_pair = {
                        "prompt": formatted_prompt_anchor,
                        "chosen": chosen_response,
                        "rejected": rejected_response
                    }
                    DPO_FORMAT_PAIR.append(dpo_pair)
                else:
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading file: {e.with_traceback()}")
    return DPO_FORMAT_PAIR

def convert_to_dataset(pair_list):
    dataset = Dataset.from_list(pair_list)
    return dataset

def save_dataset_locally(json_list, dataset, save_path):
    # save the json as well for the weighted random sampling
    with open(f"{save_path}.json", "w") as file:
        json.dump(json_list, file, indent=4)
    
    # save the hf dataset
    dataset.save_to_disk(save_path)

def main():
    # load jsonl file
    load_jsonl_file("dpo_pairs_2.jsonl")
    # convert to dataset
    dpo_dataset = convert_to_dataset(DPO_FORMAT_PAIR)
    sft_dataset = convert_to_dataset(SFT_FORMAT_PAIR)
    # save dataset locally
    save_dataset_locally(DPO_FORMAT_PAIR, dpo_dataset, "dpo_dataset_local")
    save_dataset_locally(SFT_FORMAT_PAIR, sft_dataset, "sft_dataset_local")

if __name__ == "__main__":
    main()