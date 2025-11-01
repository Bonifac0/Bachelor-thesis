import json
import re
import os

FASTA_PATH = "datasets/Pfam-A.fasta"
UNIREF_PATH = "datasets/uniref50.fasta"
TEMP_PATH = "datasets/200617_TEMPURA.json"

if not os.path.isfile(FASTA_PATH):
    raise FileNotFoundError(f"Dataset file '{FASTA_PATH}' does not exist.")
if not os.path.isfile(UNIREF_PATH):
    raise FileNotFoundError(f"Dataset file '{UNIREF_PATH}' does not exist.")
if not os.path.isfile(TEMP_PATH):
    raise FileNotFoundError(f"Dataset file '{TEMP_PATH}' does not exist.")

prot_temp_path = "datasets/prot_temp.json"
output_path = "datasets/processed_dataset.json"


def get_uniref(tempura: dict, save_sequence=False):
    # >UniRef50_UPI002E2621C6 uncharacterized protein LOC134193701 n=1 Tax=Corticium candelabrum TaxID=121492 RepID=UPI002E2621C6
    print()
    with open(UNIREF_PATH, "r") as orgprot_file:
        genes: dict = {}
        counter = 0

        for line in orgprot_file:
            if line.startswith(">"):
                counter += 1
                if counter % 100000 == 0:
                    print(f"Processed {counter / 1000000}M / 70.1M uniref", end="\r")

                pattern = r"UniRef50_(\S+).*?Tax=(.*?) TaxID=(\d+)"
                match = re.search(pattern, line)

                if match and int(match.group(3)) in tempura:
                    last_prot_id = match.group(1)  # UPI002E2621C6
                    genes[last_prot_id] = {}
                    genes[last_prot_id]["temp"] = tempura[int(match.group(3))][0]
                    genes[last_prot_id]["org"] = tempura[int(match.group(3))][1]
                    genes[last_prot_id]["org_id"] = int(match.group(3))

                else:  # Probably not known organism
                    last_prot_id = None

            elif last_prot_id and save_sequence:
                if "sequence" not in genes[last_prot_id]:
                    genes[last_prot_id]["sequence"] = line.strip()
                else:
                    genes[last_prot_id]["sequence"] += line.strip()
    print()
    return genes


def get_temp():
    with open(TEMP_PATH, "r") as temp_file:
        orgs: dict = {}
        for instance in json.load(temp_file):
            orgs[instance["taxonomy_id"]] = (
                instance["Topt_ave"],
                instance["genus_and_species"],
            )
        return orgs


def get_Pfam(prot_temp: dict):
    # >A0A671U9Z5_SPAAU/41-288 A0A671U9Z5.1 PF00001.26;7tm_1;
    print()
    output: dict = {}
    counter = 0
    with open(FASTA_PATH, "r") as fasta:
        for line in fasta:
            if line.startswith(">"):
                counter += 1
                if counter % 100000 == 0:
                    print(f"Processed {counter / 1000000}M / 57.5M Pfam", end="\r")

                data = line.split()
                fami_id = data[2].split(".")[0]
                prot_id = data[1].split(".")[0]
                if fami_id not in output:
                    output[fami_id] = {}

                if prot_id in prot_temp:
                    output[fami_id][prot_id] = prot_temp[prot_id]

            else:
                if prot_id in prot_temp:
                    if "domain" not in output[fami_id][prot_id]:
                        output[fami_id][prot_id]["domain"] = line.strip()
                    else:
                        output[fami_id][prot_id]["domain"] += line.strip()
    print()
    return output


def main():
    if os.path.exists(prot_temp_path):
        print(f"Loading prot-temp from {prot_temp_path}")
        with open(prot_temp_path, "r") as f:
            org = json.load(f)
    else:
        org = get_uniref(get_temp(), save_sequence=True)
        print(f"Writing prot-temp to {prot_temp_path}")
        with open(prot_temp_path, "w") as f:
            json.dump(org, f, indent=4)

    fami = get_Pfam(org)

    print("Removing keys with no value")
    fami = {k: v for k, v in fami.items() if v}

    print(f"Saving processed databese to {output_path}")
    with open(output_path, "w") as output:
        json.dump(fami, output, indent=4)


if __name__ == "__main__":
    main()
