import json

INPUT_PATH = "datasets/processed_dataset.json"
OUTPUT_PATH = "datasets/clear_classes.json"


def classify_temp(temp: float) -> str | None:
    if temp < 15:
        return "psychrophilic"
    elif 30 <= temp < 35:
        return "mesophilic"
    elif 50 <= temp < 70:
        return "thermophilic"
    elif temp >= 80:
        return "hyperthermophilic"
    else:
        return None


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


with open(INPUT_PATH, "r") as f:
    data = json.load(f)

stats = {
    "psychrophilic": 0,
    "mesophilic": 0,
    "thermophilic": 0,
    "hyperthermophilic": 0,
}

output = {}

cls_dropped = {
    "psychrophilic": 0,
    "mesophilic": 0,
    "thermophilic": 0,
    "hyperthermophilic": 0,
}

for fam_id, fam in data.items():
    for prot_id, prot in fam.items():
        cls = classify_temp(prot["temp"])
        if cls:
            stats[cls] += 1
            if fam_id not in output:
                output[fam_id] = {}
            output[fam_id][prot_id] = prot
        else:  # if in temperature gap
            wide_cls = classify_temp_nogaps(prot["temp"])
            if wide_cls:
                cls_dropped[wide_cls] += 1
            else:
                print("wtf")

print("Dropped because temperature gaps")
print(cls_dropped)

print("What remained:")
print(stats)
print(f"sum: {sum([i for i in stats.values()])}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=4)
