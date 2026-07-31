from typing import final
import torch
from torch.utils.data import WeightedRandomSampler
import random
from datasets import Dataset
import json
import numpy as np

raw_data = []
with open("dpo_pairs_2.jsonl", "r") as f:
    for line in f:
        if line.strip():  # Skip empty lines
            raw_data.append(json.loads(line))

categories = [item["cuad_category"] for item in raw_data]
unique_categories, counts = np.unique(categories, return_counts=True)
class_counts = dict(zip(unique_categories, counts))

# Compute inverse frequency weights: weight = total_samples / (num_classes * class_count)
total_samples = len(raw_data)
num_classes = len(unique_categories)
class_weights = {cat: total_samples / (num_classes * count) for cat, count in class_counts.items()}

print(f"Dataset Distribution: {class_counts}")
print(f"Calculated Weights: {class_weights}")

# 3. Map weights to every individual sample
sample_weights = [class_weights[item["cuad_category"]] for item in raw_data]

# 4. Create the WeightedRandomSampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=total_samples, # Match original size, or adjust as needed
    replacement=True           # True allows minority classes to be oversampled
)

# 5. Generate balanced indices and create the balanced dataset
# We draw from the sampler once to create a static, balanced offline dataset
balanced_indices = list(sampler)
balanced_raw_data = [raw_data[i] for i in balanced_indices]

# Verify the new distribution
balanced_categories = [item["cuad_category"] for item in balanced_raw_data]
new_categories, new_counts = np.unique(balanced_categories, return_counts=True)
print(f"Balanced Dataset Distribution: {dict(zip(new_categories, new_counts))}")

# 6. Convert to Hugging Face Dataset format for DPOTrainer
# Ensure your JSON keys match DPO requirements: "prompt", "chosen", "rejected"
final_dataset = Dataset.from_list(balanced_raw_data)

final_dataset.save_to_disk("balanced_dataset_post_weighted_sampling")