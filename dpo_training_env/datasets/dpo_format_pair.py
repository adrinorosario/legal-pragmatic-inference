import json
import traceback
import pyarrow as pa
from datasets import Dataset, load_from_disk

PROMPT_TEMPLATE = (
    "Given the following contractual clause which falls in the category of {cuad_category}, provide judicial reasoning of "
    "the kind a court would apply when interpreting clauses that raise "
    "similar legal questions:\n\n{prompt_anchor}"
)

def load_jsonl_file(file_path):
    """Loads JSONL and returns localized lists to avoid global state contamination."""
    dpo_pairs = []
    sft_pairs = []
    
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                full_data = json.loads(line)

                # Extract required keys
                prompt_anchor = full_data["prompt_anchor"]
                chosen_response = full_data["chosen"]
                rejected_response = full_data["rejected"]
                category_anchor = full_data["cuad_category"]

                # Format prompt
                formatted_prompt_anchor = PROMPT_TEMPLATE.format(
                    cuad_category=category_anchor,
                    prompt_anchor=prompt_anchor
                )

                # Append structurally clean dictionaries
                sft_pairs.append({
                    "prompt": formatted_prompt_anchor,
                    "response": chosen_response
                })

                dpo_pairs.append({
                    "prompt": formatted_prompt_anchor,
                    "chosen": chosen_response,
                    "rejected": rejected_response
                })
                
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print("Error loading file:")
        traceback.print_exc()  # Fixed the buggy e.with_traceback() call
        
    return dpo_pairs, sft_pairs

def load_balanced_dataset(load_path="balanced_dataset_post_weighted_sampling"):
    """Loads a weighted-sampled, balanced Dataset from disk and reformats it
    into DPO/SFT pairs — mirrors load_jsonl_file's transform logic."""
    dpo_pairs = []
    sft_pairs = []

    try:
        dataset = load_from_disk(load_path)
    except FileNotFoundError:
        print(f"Dataset not found at: {load_path}")
        return dpo_pairs, sft_pairs
    except Exception:
        print("Error loading dataset from disk:")
        traceback.print_exc()
        return dpo_pairs, sft_pairs

    for row in dataset:
        try:
            prompt_anchor = row["prompt_anchor"]
            chosen_response = row["chosen"]
            rejected_response = row["rejected"]
            category_anchor = row["cuad_category"]
        except KeyError as e:
            print(f"Skipping row, missing key: {e}")
            continue

        formatted_prompt_anchor = PROMPT_TEMPLATE.format(
            cuad_category=category_anchor,
            prompt_anchor=prompt_anchor
        )

        sft_pairs.append({
            "prompt": formatted_prompt_anchor,
            "response": chosen_response
        })

        dpo_pairs.append({
            "prompt": formatted_prompt_anchor,
            "chosen": chosen_response,
            "rejected": rejected_response
        })

    return dpo_pairs, sft_pairs

def save_dataset_locally(json_list, save_path):
    """Compiles and writes datasets using PyArrow to completely isolate Python 3.14 pickle bugs."""
    # 1. Save standard JSON backup for weighted random sampling
    with open(f"{save_path}.json", "w") as file:
        json.dump(json_list, file, indent=4)
    
    # 2. Safely construct a PyArrow Table directly from raw Python types
    arrow_table = pa.Table.from_pylist(json_list)
    
    # 3. Instantiate the Dataset wrapper safely using a pre-built structural table 
    # and immediately commit to disk before fingerprint loops can trigger.
    dataset = Dataset(arrow_table)
    dataset.save_to_disk(save_path)
    print(f"Successfully compiled and saved dataset to: {save_path}")

def main():
    # Load separate list returns cleanly
    # dpo_list, sft_list = load_jsonl_file("dpo_pairs_2.jsonl")
    dpo_list, sft_list = load_balanced_dataset("balanced_dataset_post_weighted_sampling")

    if not dpo_list or not sft_list:
        print("Data compilation aborted: Lists are empty.")
        return

    # save_dataset_locally(dpo_list, "dpo_dataset_balanced")
    save_dataset_locally(sft_list, "sft_dataset_balanced")
    
    # if not dpo_list or not sft_list:
    #     print("Data compilation aborted: Lists are empty.")
    #     return

    # Process and save cleanly via explicit PyArrow tables
    # save_dataset_locally(dpo_list, "dpo_dataset_revised")
    # save_dataset_locally(sft_list, "sft_dataset_revised")

if __name__ == "__main__":
    main()
