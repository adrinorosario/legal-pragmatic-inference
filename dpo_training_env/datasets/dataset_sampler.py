import torch
from torch.utils.data import WeightedRandomSampler
import random
from datasets import Dataset
import json

with open("dpo_pairs_2.jsonl", "r") as file:
    raw_data = json.load(file)

categories = set(item["cuad_category"] for item in raw_data)