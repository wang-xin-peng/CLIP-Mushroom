import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path("data/FungiTastic/FungiTastic-Mini")
META_ROOT = Path("data/FungiTastic/metadata/FungiTastic-Mini")


def get_metadata_splits():
    """Load train/val/test metadata and return DataFrames with image paths.

    Returns:
        train_df, val_df, test_df: DataFrames with columns including
            scientificName, filename, image_path
    """
    splits = {}
    for name in ["Train", "Test", "Val"]:
        df = pd.read_csv(META_ROOT / f"FungiTastic-Mini-{name}.csv")
        df["image_path"] = df["filename"].apply(
            lambda x: str(DATA_ROOT / name.lower() / "300p" / x) if pd.notna(x) else ""
        )
        df["label"] = df["scientificName"].fillna("unknown")
        # Drop rows without valid images
        df = df[df["filename"].notna() & (df["filename"] != "")]
        splits[name.lower()] = df
    return splits["train"], splits["val"], splits["test"]


def split_species(train_df, unseen_ratio=0.2, random_seed=42):
    """Split species into seen (for Linear Probe) and unseen (for zero-shot test).

    From all species in the dataset, randomly select `unseen_ratio` as unseen.
    Unseen species are excluded from Linear Probe training but included in zero-shot evaluation.

    Returns:
        seen_species: set of species names for training
        unseen_species: set of species names held out for zero-shot only
    """
    all_species = sorted(train_df["label"].unique())
    rng = np.random.RandomState(random_seed)
    n_unseen = max(1, int(len(all_species) * unseen_ratio))
    unseen_species = set(rng.choice(all_species, n_unseen, replace=False))
    seen_species = set(all_species) - unseen_species
    return seen_species, unseen_species
