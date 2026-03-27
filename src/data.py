import os

import pandas as pd

# Default path to the datasets directory at the repo root
DEFAULT_DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
)

# All datasets from the paper (curated + uncurated)
ALL_DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
    "cities_cities_conj",
    "cities_cities_disj",
    "companies_true_false",
    "common_claim_true_false",
    "counterfact_true_false",
    "likely",
]


def load_dataset(
    dataset_name: str,
    datasets_dir: str = DEFAULT_DATASETS_DIR,
) -> tuple[list[str], list[int]]:
    """
    Load statements and binary labels from a dataset CSV.

    Returns:
        statements: list of statement strings
        labels:     list of int labels (1 = true, 0 = false)
    """
    path = os.path.join(datasets_dir, f"{dataset_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at '{path}'.\n"
            f"Available: {ALL_DATASETS}"
        )
    df = pd.read_csv(path)
    statements = df["statement"].tolist()
    labels = df["label"].astype(int).tolist()
    return statements, labels
