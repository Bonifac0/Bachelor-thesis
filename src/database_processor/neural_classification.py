import json
import os
import argparse
import time
from src.predictor import Classificator

"""
to run:
python -m src.database_processor.neural_classification inp.json out.json

because pyhon need to load Clasificator
"""


def pars_arguments():
    parser = argparse.ArgumentParser(
        description="Filter protein data by temperature categories."
    )
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("output", help="Output JSON file path")
    args = parser.parse_args()

    in_file = args.input
    out_file = args.output

    # Check that input file exists
    if not os.path.isfile(in_file):
        raise FileNotFoundError(f"Input file '{in_file}' does not exist.")
    return in_file, out_file


def collect_proteins(data: dict) -> tuple[list, list]:
    """
    Return list of proteins and their place in families.
    Also dropp proteins that are too long.
    """
    dom_dropped = 0
    sec_dropped = 0

    protein_list = []
    protein_keys = []

    # Collect all proteins
    for fam, entries in data.items():
        for prot_id, entry in entries.items():
            if "sequence" in entry:
                if len(entry["sequence"]) > 5000:
                    sec_dropped += 1
                    continue
                if len(entry["domain"]) > 4000 and False:  # turned off
                    dom_dropped += 1
                    continue
                protein_list.append((prot_id, entry["sequence"]))
                protein_keys.append((fam, prot_id))

    print(f"Dropped becaouse domain len: {dom_dropped}")
    print(f"Dropped becaouse sequence len: {sec_dropped}")
    return protein_list, protein_keys


def print_eta(start_time, current_batch, total_batches):
    elapsed = time.time() - start_time
    avg_time = elapsed / current_batch
    remaining_batches = total_batches - current_batch
    eta_seconds = remaining_batches * avg_time

    # Format ETA into hh:mm:ss
    hrs, rem = divmod(int(eta_seconds), 3600)
    mins, secs = divmod(rem, 60)
    eta_formatted = f"{hrs:02}:{mins:02}:{secs:02}"
    return f" | ETA: {eta_formatted}"


def main():
    classificator = Classificator()

    with open(IN_FILE, "r") as f:
        data = json.load(f)

    protein_list, protein_keys = collect_proteins(data)

    # Process in batches with simple print feedback
    preds = []
    total_batches = (len(protein_list) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    print()
    for batch_idx in range(0, len(protein_list), BATCH_SIZE):
        batch = protein_list[batch_idx : batch_idx + BATCH_SIZE]
        batch_number = batch_idx // BATCH_SIZE + 1

        print(
            f"Processing batch {batch_number}/{total_batches} ({len(batch)} proteins) {print_eta(start_time, batch_number, total_batches)}",
            end="\r",
        )

        outputs = classificator.classify(batch)
        preds.extend(outputs)

    print()

    # Assign predictions back to data
    for (fam, prot_id), pred in zip(protein_keys, preds):
        data[fam][prot_id]["pred"] = pred

    print(f"Saving to {OUT_FILE}")
    # Save results
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # BATCH_SIZE = 32
    BATCH_SIZE = 2
    IN_FILE, OUT_FILE = pars_arguments()
    main()
