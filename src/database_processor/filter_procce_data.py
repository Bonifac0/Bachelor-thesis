import json
import os
import random

INPUT_PATH = "datasets/processed_dataset.json"
OUTPUT_PATH = "test.json"

# Define temperature categories
CATEGORIES = {
    "lt_20": lambda t: t is not None and t < 20,
    "20_45": lambda t: t is not None and 20 <= t < 45,
    "45_80": lambda t: t is not None and 45 <= t < 80,
    "gt_80": lambda t: t is not None and t >= 80,
}
CATEGORY_LIMIT = 100

if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(f"Input file '{INPUT_PATH}' does not exist.")

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# Collect entries by category
category_entries = {cat: [] for cat in CATEGORIES}
for fam, entries in data.items():
    for prot_id, entry in entries.items():
        temp = entry.get("temp")
        for cat, cond in CATEGORIES.items():
            if cond(temp):
                category_entries[cat].append((fam, prot_id, entry))
                break

# Randomly select up to 50 from each category
selected_entries = {cat: random.sample(entries, min(CATEGORY_LIMIT, len(entries)))
                    for cat, entries in category_entries.items()}

# Build output structure
output_data = {}
for cat_entries in selected_entries.values():
    for fam, prot_id, entry in cat_entries:
        if fam not in output_data:
            output_data[fam] = {}
        output_data[fam][prot_id] = entry

with open(OUTPUT_PATH, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Filtered data written to {OUTPUT_PATH} with up to {CATEGORY_LIMIT} entries per category.")
