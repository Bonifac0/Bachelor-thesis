from predictor import Classificator


def main():
    classificator = Classificator(MODEL_PATH)

    protein_list = [
        ("A0A1M6DL67", "MDRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAGRWLRVNAYLRTAVPGRRR") * 2
    ]

    # Process in batches with simple print feedback
    preds = []
    total_batches = (len(protein_list) + BATCH_SIZE - 1) // BATCH_SIZE

    print()
    for batch_idx in range(0, len(protein_list), BATCH_SIZE):
        batch = protein_list[batch_idx : batch_idx + BATCH_SIZE]
        batch_number = batch_idx // BATCH_SIZE + 1

        print(
            f"Processing batch {batch_number}/{total_batches} ({len(batch)} proteins)",
            end="\r",
        )

        outputs = classificator.classify(batch)
        preds.extend(outputs)

    print()


if __name__ == "__main__":
    # BATCH_SIZE = 32
    BATCH_SIZE = 2
    MODEL_PATH = "resources/model-664.pt"  # .pt file
    main()
