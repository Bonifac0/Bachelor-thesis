import json
from src.heplers.levenshtein import levenshtein

"""
to run:
python -m src.training.filter_test_dataset.py
"""

INPUT_PATH = "datasets/processed_dataset.json"
# OUTPUT_PATH = "datasets/testing_dataset.json"
OUTPUT_PATH = "test.json"


def classify_temp_nogaps(temp: float) -> str | None:
    if temp < 20:
        return "psychrophilic"
    elif 20 <= temp < 45:
        return "mesophilic"
    elif 45 <= temp < 80:
        return "thermophilic"
    elif temp >= 80:
        return "hyperthermophilic"
    else:
        return None


def extract_families(input_json: dict) -> dict:
    # output will store only families that meet the pair condition
    output = {}

    # these are the required class pairings
    valid_pairs = {
        ("psychrophilic", "hyperthermophilic"),
        # ("psychrophilic", "thermophilic"),
        # ("mesophilic", "hyperthermophilic"),
    }

    for fam_id, proteins in input_json.items():
        # gather proteins by class
        class_map = {}
        for prot_id, info in proteins.items():
            cls = classify_temp_nogaps(info["temp"])
            if cls is None:
                continue
            class_map.setdefault(cls, {})[prot_id] = info

        # check if family contains one of the required combinations
        classes_present = set(class_map.keys())
        triggered = any(
            a in classes_present and b in classes_present for (a, b) in valid_pairs
        )

        if triggered:
            # include only the proteins belonging to the relevant classes
            selected = {}

            needed_classes = set()
            for a, b in valid_pairs:
                if a in classes_present and b in classes_present:
                    needed_classes.update([a, b])

            # flatten selected proteins
            for cls in needed_classes:
                for pid, info in class_map[cls].items():
                    selected[pid] = info

            output[fam_id] = selected

    return output


def extract_pairs(input_json: dict, similarity_threshold: float = 0.5):
    """
    similarity_threshold: between 0 and 1
    similarity = 1 - (levenshtein / max_len)
    """
    valid_pairs = {
        ("psychrophilic", "hyperthermophilic"),
        ("psychrophilic", "thermophilic"),
        # ("mesophilic", "hyperthermophilic"),
    }

    output = []

    for fam_id, proteins in input_json.items():
        print("s")
        # Group proteins by class
        class_map = {}
        for pid, info in proteins.items():
            cls = classify_temp_nogaps(info["temp"])
            if cls is None:
                continue
            class_map.setdefault(cls, {})[pid] = info

        # For each allowed class pairing, check combinations
        for cold_cls, warm_cls in valid_pairs:
            if cold_cls not in class_map or warm_cls not in class_map:
                continue

            cold_group = class_map[cold_cls]
            warm_group = class_map[warm_cls]

            for cid, cinfo in cold_group.items():
                for wid, winfo in warm_group.items():
                    d1 = cinfo["domain"]
                    d2 = winfo["domain"]

                    # Compute similarity
                    dist = levenshtein(d1, d2)
                    max_len = max(len(d1), len(d2))
                    similarity = 1 - dist / max_len if max_len > 0 else 0

                    if similarity >= similarity_threshold:
                        # append tuple of cold protein info, warm protein info
                        output.append((cinfo, winfo))

    return output


with open(INPUT_PATH, "r") as f:
    data = json.load(f)

output = extract_pairs(data)
# output = extract_families(data)

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=4)
