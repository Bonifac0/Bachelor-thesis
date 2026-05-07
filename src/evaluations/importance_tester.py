import random
from src.helpers.importance_vis import make_importance_general, make_importance_diff
from src.training.run_model import ModelRunner
from src.helpers.captum_attribution import get_captum_attribution
from src.helpers.print_eta import ETA
import numpy as np


"""
python -m src.evaluations.importance_tester

main script for testing and comparing importance prediktor with sum method
"""


def random_single_mutations(protein):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mutated_sequences = []

    for i, aa in enumerate(protein):
        choices = [x for x in amino_acids if x != aa]
        new_aa = random.choice(choices)
        mutated_sequences.append(protein[:i] + new_aa + protein[i + 1 :])

    return mutated_sequences


def single_revert(baseline: str, modified: str) -> tuple[list[str], list[int]]:
    """
    Returns list of mutants with one less mutated residum and indices of changes
    """
    if len(baseline) != len(modified):
        raise ValueError("Sequences must be the same length")

    diffs = [i for i in range(len(baseline)) if baseline[i] != modified[i]]
    variants = []

    for i in diffs:
        seq = list(modified)
        seq[i] = baseline[i]
        variants.append("".join(seq))

    return variants, diffs


def generate_random_mutations(dna: str, value: float, num_mutations: int = 10):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    length = len(dna)
    num_changes = max(1, int(length * value))

    mutations = []

    for _ in range(num_mutations):
        dna_list = list(dna)
        positions = random.sample(range(length), num_changes)

        for pos in positions:
            original = dna_list[pos]
            choices = [aa for aa in amino_acids if aa != original]
            dna_list[pos] = random.choice(choices)

        mutations.append("".join(dna_list))

    return mutations


def use_chaotic_mutations(
    classificator, baseline, baseline_score, iterations=50
) -> np.ndarray:
    percetage = 0.1
    mutations = generate_random_mutations(baseline, percetage, iterations)
    data = []
    for random_mutant in mutations:
        if single_revert is None:
            print("didnt make it")
            continue
        variants, indices = single_revert(random_mutant, baseline)

        mutations = [("", mut) for mut in variants]
        probability = [i[3] for i in classificator.classify(mutations)]

        counter = 0
        indices.append(-1)

        single_data = []
        # print(indices)
        for idx, mut in enumerate(baseline):
            if idx == indices[counter]:
                score = (
                    baseline_score - probability[counter]
                ) / percetage  # normalization
                counter += 1
            else:
                score = 0
            single_data.append(score)
        # print(single_data)
        data.append(single_data)

    real_score = np.mean(data, axis=0)
    # print()
    # print(real_score)
    return real_score


def aggregate_attribution(ig_attribution: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L1 aggregation for an Integrated Gradients embedding attribution.
    """
    # TODO read https://arxiv.org/html/2507.18043v1?utm_source=chatgpt.com
    # source acording to chatbot

    l1 = np.abs(ig_attribution).sum(axis=-1)
    return l1 / (l1.max() + eps)


def aggregate_log_sigmoid(attribution):
    s = np.abs(attribution).sum(axis=-1)
    NORM_MEDIAN = -0.275390625
    steepnes = 18
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - NORM_MEDIAN)))


def main():
    runner = ModelRunner("2HL_64_16")

    proteins = [
        # {
        #     "prot_id": "alice",
        #     "domain": "DRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAG",
        #     "mutant": "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG",
        # },
        # {
        #     "prot_id": "bob",
        #     "domain": "ALQLRAETGAATPADWHWGDVAIIADNRTEADVIRQFRA",
        #     "mutant": "AYFLRAETGAATPNKWPWGDVAIIADVRMEDDVIKKFRA",
        # },
        # {
        #     "prot_id": "cyril",
        #     "domain": "FLQLEDIHPSAVEALKQDGYHQTETHTHAPIGAELIKAIGNAHFVGLRSRTRLTEEVQSQAAKLTAIGCFCIGTNQVDLPAANGSSAEAASDAHIR",
        #     "mutant": "FLGLEDIPPEAVEALKMYGYFQTETHTHAPIGAELIKAIFMAHFVGKRSRTRLTEEVQSEAAKLTAIGCFCIGTNQVNLPAANGSSAERASDAHVR",
        # },
        # {
        #     "prot_id": "dave",
        #     "domain": "VVSLDKISDRMKSLIRSKLHADFEIVFCENDRDVHNHISSANVLITFTRGISKEWMEQAETCRFIQKLGAGVNNIDLETASNRGIPVSMTKGGNARSVAEHAVALMMMVFKQMNIAHNEIVNKGTHSRCRSRCVRAGADRTRTSFYYIDQYGAHAAHR",
        #     "mutant": "VVSLDKISRRMKSLIRSYLHKDFEIVFCENDRDVHNHISSANVLITFTWGISKEWIEQAETCIFIQKLGAGVNNIDLEVASERGPPVSMTKVGFARSVAEHAVALMMMVFKQMNIAWREIVNKGTHSRCRSRCVRAGADRTRTSFYYIDQYGAHARHR",
        # },
        # {
        #     "prot_id": "emil",
        #     "domain": "LLVAYPTRPRQMALLAEAYTIHRLDLAEDKVAMLVEVGPRCTAMLCNGHVTIDEAFLAQVPNLRIAASSSVGYDTIDVPALTRAGVRLTNTPDVLTDDVADTA",
        #     "mutant": "LLVAYPTRPRQMALLAEAYTIHVLDLAEFKVAMLVVVGFRCTAMLCNGHVTDKEAFLWIVPNLRIAASSSVGYDTIDVWALTRAGVRLYNIPDVLTDDVADTA",
        # },
        # {
        #     "prot_id": "fanda",
        #     "domain": "MIKVISRYCVSYDNVDIEAVKDLRILVTSSAVGYVIIIAEHTIN",
        #     "mutant": "MIKVISRYTVSYDNVPIEAVKDLRILVTSSAVGYVIIIAEHTIN",
        # },
        # {
        #     "prot_id": "R2D2",
        #     "domain": "ILYFDHLPDDLKQIVLEKVQPGVTLTFWSDLSERQRELELRQTDYFILTAFKIDEKIIRLAEKLKFIQKVGIGVDNIDLVAAASRSVPVSNTPGGNSISVSEATILFILALYRRLIEMNQATKEGKWYSWNFRSSSYEIYGKTYGLIGMGNIGYETAKRSKAFGTNIVYYDKRRLSFEKEKEIHAQYVSLDQLILFILALYRRLIEMNQATKEGKWYSWNFRSSSYEIYGKTYGLIGMGNIGYETAKRSKAFGTNIVYYDKRRLSFEKEKEIHAQYVSLDQLLQVSDIISVHVPLLPETRV",
        #     "mutant": "ILYFDSLPDDLKQIVLEKVDPGVTLFFWSDLREWTRELELRQTDYFILTAFKIDEKIIRLAEKLKFIEKVGIGVWNEDLVAALSRSVYVINTPGGNSISVSEATILFILALYRRLIEMNQATKEGKWYSWNFRSSSYEIYGKTYGLIGMGNIGYETAKRSKAFGTNIVYYDKRRLSFEKEKEIHAQYVSLDQLILFILALYRRLIEMFQATKEGKWYSWNFRSSSYEIYGKTYGLIGMGNIGYETAKRSKAFGTNIVYYDKRRLSFEKEKEIHAQYVSLDQLLQVSDIISVHVPLLPETRY",
        # },
        {
            "prot_id": "diff_A0A853G6I6",
            "domain": "AWAVLNQQRQFKAYAAQQTEGRWLRHPMQSAGEHCVGIAGQGQLGAACAGTLLELGFQVRGWARDIKLNMPAALTQFAGPESLEDFLSGCDTLVCLLPLTPDTQGFIGERVLRQLPPHAHVINASRGEHVDQDALLSALDTGLLAHATLDVFHEEPLPPEHPYWKHPRITVTPHVG",
            "mutant": "AWAVLNKWMQFKYYAAQITEGRVLRHLMQSAWEHCVGIAGYGILGAWCAWTLLELGFQVRGWARDIKLRMPAWWVQFWGPESLEDFLSGCDTLVCLLPLTPDTQGFIGERVLRVLPPHAHVINASRGEHVDKDALLSALRKGLLAHATLDVFHEEYLPPEHPYWKHPRITVTPHVG",
        },
        # {
        #     "prot_id": "diff_A0A839S096",
        #     "domain": "MDELPGAADAVSLHAPLTAETSKLLGAAAFERMRRGAFLVNTSRGGLVDQDALADAVGFTGSTGGGSKLLGAAAFERMRRGAFLVNTSRGGLVDQDALADAVGFTGSTGGGRALYDLAVGRENPIPFYGELGSINPVVVTEAVVAARGREVAEGLAGSFRLGTGQFCTKPGVVLVPADSGFDTQVADAVRPSEPAPMLTARMRDAFQAG",
        #     "mutant": "MDELPGAADAVLLHYPLTKETSKLLGAAACEKMRRGAFLVNTSRGGLVDQDALADAVGFTGSTGGGSKLLGAAAFERMRRGAFLVNTSRGGLVDQDALADAVGETGSTGGGRALYDLVVGREPPIPFYKELGSIPNVVVTEAVVAARGREVAEGLAGSFRLGTGFFLTKPGVVLVPADSGFDTQVADAVRPSEVLPMLTAYMRRAFQAG",
        # },
        # {
        #     "prot_id": "diff_A0A7W0HNS6",
        #     "domain": "MGVLLATGKRLPHYWDHQRAGTWDRSGNLSFGDGPAPRHLDESTVLIIGYGSIGALLESRLAPFGTRILRIARHPRDGVGGPEALPESLGQADAVVLLAPSTPETRGMVGEKFLAAMRPGALLVNAARGALVDHDALLVALRAGQVRAALDATEPEPLPDGHPLWSAPGIMITPHVA",
        #     "mutant": "MGVLGATGKRLPHYWMHFRYGTWDRSGNLSFGDGWAPRHLDEITVLIIGYGSIGAYLESRLAPFGTRILRIARHPRDGVYYPEALFESLGQADAVVLLAPSTPETRGMVGEKFLAAMRPGALLVNAARGAYVDPRALLVALRAGFVRAALDATEPEYLPDGHWLWSAPGIVITPHVA",
        # },
        # {
        #     "prot_id": "diff_A0A7Y9IVL0",
        #     "domain": "IAGILALARNVPQWNATQQARGWLPLDGPLTPRDLPGQHAVIVGMGPIGLRVAEVLRALGMTVSGVRRHAEPLQGFEEVVGFNEFDRVVAAADWLVLTCPLTSLTSRLVGREVFDRLPAGARLINVGRGGVVDEAALLDALRTGRLAGAYSDVFETEPLPADAPLRDAPNLIISSHTA",
        #     "mutant": "IAGILALARNVPQWNALLIARGWLPLDGPLTPRDLPGMHAVIVGMGPIGLRVAEVLRALGMTVSGVRRHNEPLQGFEEVVYFNEFDRVVAAYDWLVLTCPKTSLTSRLVGREVFDRLPAGALLINVGRGYVVDENALLDALRTGRLAGYYVDVFETEPLPNDAPLRDIPNLIIISHTA",
        # },
        # {
        #     "prot_id": "diff_A0A840UMS6",
        #     "domain": "CKLLHLKKSIIADNSECTSRGFATALKNKIKNSDKKKITILGAGPVGQMAAKNLKADDIEIIIYDIDKEKMDTAANNLGIGTAKSLQEALDKSNFYFEATTSINTIKENDIKADTFIDIFHISEQLTKYDSDLRRIIDMGLNELGAAAAGANLTELQKYIKGECVGIVPITSGEGIINSFSQTIKSIIDFTGLTSFITNATDVAGIAEAVERKATMLFIADDATCKLLHLKKSIIA",
        #     "mutant": "CKLLHLKKSIIADNYECTSRGFATALINKIKNSDKKKIWILGAGPVGLMAAKNLKHDDEEIIIYDIDKEKMKTAANNLGIGTAKSLQEALDKINFYFEATTSINTIKENDIKPDTFIDIFHISEQLTKYDSDLRRIIDMFLNELGAAAAGANLTELQKYIKGECVGIVPITSGEGICNYFSQTIKSIIDFTILTSFITNATDVAGIAEAVERKATMLFIADDATCKLLHLKCSIIA",
        # },
        # {
        #     "prot_id": "coldshock",
        #     "domain": "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA",
        #     "mutant": "MQRGKVKWFNNEKGYGFIEVEGGSDVFVHFTAIQGEGFKTLEEGQEVSFEIVQGNRGPQAANVVKL-",
        # },
    ]

    protein_count = len(proteins)
    eta = ETA(protein_count)
    print()
    print(
        f"Tested protein {0}/{protein_count} | ETA: ?",
        end="\r",
    )

    counter_all = 0
    pred_correct = 0
    agg_correct = 0

    for idx, protein in enumerate(proteins):
        probability = (
            runner.classificator.classify([("", protein["domain"])])[0][3],
            runner.classificator.classify([("", protein["mutant"])])[0][3],
        )

        mut_attribution = get_captum_attribution(
            runner.classificator, protein["mutant"]
        )
        dom_attribution = get_captum_attribution(
            runner.classificator, protein["domain"]
        )
        pred_mut: np.ndarray = runner.predict_importance(
            protein["mutant"], mut_attribution
        )
        pred_dom: np.ndarray = runner.predict_importance(
            protein["domain"], dom_attribution
        )

        # real_decrease: np.ndarray = use_chaotic_mutations(
        #     classificator, protein["mutant"], probability[1]
        # )

        aggrt_mut = aggregate_log_sigmoid(mut_attribution)
        aggrt_dom = aggregate_log_sigmoid(dom_attribution)

        for j, (d, m) in enumerate(zip(protein["domain"], protein["mutant"])):
            if d != m:
                counter_all += 1
                if pred_dom[j] < 0.5 and pred_mut[j] > 0.5:
                    pred_correct += 1
                if aggrt_dom[j] < 0.5 and aggrt_mut[j] > 0.5:
                    agg_correct += 1

        data = np.row_stack(
            [
                pred_mut,
                pred_dom,
                aggrt_mut,
                aggrt_dom,
                # real_decrease,
            ]
        )

        labels = [
            "Predictor mutant",
            "Predictor domain",
            "Captum relative mutant",
            "Captum relative domain",
            # "Real decrease",
        ]

        make_importance_diff(
            protein, data, probability, labels, outdir="test_importance/full"
        )

        data_only_mut = np.row_stack(
            [
                pred_mut,
                aggrt_mut,
                # real_decrease,
            ]
        )

        labels_only_mut = [
            "Predictor mutant",
            "Captum relative mutant",
            # "Real decrease",
        ]
        make_importance_diff(
            protein,
            data_only_mut,
            probability,
            labels_only_mut,
            outdir="test_importance/only_mut",
        )

        data_only_pred = np.row_stack(
            [
                pred_mut,
                pred_dom,
            ]
        )

        labels_only_pred = [
            "Mutant",
            "Domain",
        ]
        make_importance_diff(
            protein,
            data_only_pred,
            probability,
            labels_only_pred,
            outdir="test_importance/only_pred",
        )

        print(
            f"Tested protein {idx + 1}/{protein_count} {eta.print_eta(idx + 1)}",
            end="\r",
        )
    print()
    eta.print_elapsed()
    print(f"Predictrion accuracy: {pred_correct / counter_all}")
    print(f"Aggregation accuracy: {agg_correct / counter_all}")


if __name__ == "__main__":
    main()
