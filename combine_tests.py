import json
import glob
import os

# Folder where your JSON files are stored
input_folder = "tests"
output_file = "test_combined.json"

# This will hold the combined data
combined_data = {}

# Keep track of duplicates
duplicates = []

# Iterate over all JSON files in the folder
for file_path in glob.glob(os.path.join(input_folder, "*.json")):
    print(f"Reading: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

        # Merge data into combined_data
        for pfam_id, entries in data.items():
            if pfam_id not in combined_data:
                combined_data[pfam_id] = {}

            for protein_id, protein_data in entries.items():
                if protein_id in combined_data[pfam_id]:
                    # Duplicate detected
                    duplicates.append((pfam_id, protein_id, file_path))
                else:
                    combined_data[pfam_id][protein_id] = protein_data

# Save the combined JSON
with open(output_file, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"\nCombined JSON saved to {output_file}")

# Report duplicates
if duplicates:
    print("\nDuplicates detected:")
    for pfam_id, protein_id, file_path in duplicates:
        print(f"  - {pfam_id} -> {protein_id} (also found in {file_path})")
else:
    print("\nNo duplicates found.")
