import json
import re

fasta_path = "datasets/Pfam-A.fasta"
orgprot_path = "datasets/uniref50.fasta"
temp_path = "datasets/200617_TEMPURA.json"

output_path = "datasets/processed_dataset.json"


def get_uniref(tempura: dict, save_sequence=False):
    # >UniRef50_UPI002E2621C6 uncharacterized protein LOC134193701 n=1 Tax=Corticium candelabrum TaxID=121492 RepID=UPI002E2621C6
    with open(orgprot_path, "r") as orgprot_file:
        genes: dict = {}
        counter = 0

        for line in orgprot_file:
            if line.startswith(">"):
                counter += 1
                if counter % 100000 == 0:
                    print(f"Processed {counter / 1000000}M uniref...")

                if counter > 2000:
                    return genes

                pattern = r"UniRef50_(\S+).*?Tax=(.*?) TaxID=(\d+)"
                match = re.search(pattern, line)

                if match:
                    last_prot_id = match.group(1)  # UPI002E2621C6
                    genes[last_prot_id] = {}
                    # genes[last_prot_id]["org"] = match.group(2)  # Corticium candelabrum
                    # genes[last_prot_id]["org_id"] = int(match.group(3))  # 121492
                    if last_prot_id in tempura:
                        genes[last_prot_id]["temp"] = tempura[int(last_prot_id)]

                else:  # Probably not known organism
                    # print(f"Error while parsing uniref position: {counter}")
                    # print(line)
                    last_prot_id = None

            elif last_prot_id and save_sequence:
                if "sequence" not in genes[last_prot_id]:
                    genes[last_prot_id]["sequence"] = line.strip()
                else:
                    genes[last_prot_id]["sequence"] += line.strip()
        return genes


def get_temp():
    with open(temp_path, "r") as temp_file:
        orgs: dict = {}
        for instance in json.load(temp_file):
            orgs[instance["taxonomy_id"]] = instance["Topt_ave"]
        return orgs


def get_Pfam(prot_temp: dict, split_lim: int):
    # >A0A671U9Z5_SPAAU/41-288 A0A671U9Z5.1 PF00001.26;7tm_1;
    output: dict = {}
    counter = 0
    with open(fasta_path, "r") as fasta:
        for line in fasta:
            if line.startswith(">"):
                counter += 1
                if counter % 100000 == 0:
                    print(f"Processed {counter / 1000000}M Pfam...")

                if counter > 2000000:
                    return output

                data = line.split()
                fami_id = data[2].split(".")[0]
                prot_id = data[1].split(".")[0]
                if fami_id not in output:
                    if counter > split_lim:
                        return output
                    output[fami_id] = {}

                if prot_id in prot_temp:
                    output[fami_id][prot_id] = prot_temp[prot_id]
                else:
                    output[fami_id][prot_id] = {"temp": "uniref missing"}

            else:
                if prot_id in prot_temp:
                    if "pfam_sec" not in output[fami_id][prot_id]:
                        output[fami_id][prot_id]["pfam_sec"] = line.strip()
                    else:
                        output[fami_id][prot_id]["pfam_sec"] += line.strip()
    return output


"""
TODO
vyresit nekompatibilni jmena organismu v org-prot a temp-org
"""


def main():
    org = get_uniref(get_temp())
    fami = get_Pfam(org, 1000000)
    with open("test.json", "w") as output:
        json.dump(fami, output, indent=4)


if __name__ == "__main__":
    main()
