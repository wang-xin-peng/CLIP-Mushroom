import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor


class FungiDataset(Dataset):
    """PyTorch Dataset for fungi images with CLIP preprocessing.

    Args:
        df: DataFrame with 'image_path' and 'label' columns
        label_map: dict mapping label string -> integer index (optional for zero-shot)
        processor: CLIPProcessor instance
    """

    def __init__(self, df, label_map=None, processor=None):
        self.df = df.reset_index(drop=True)
        self.processor = processor or CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        if label_map is not None:
            self.labels = df["label"].map(label_map).values
        else:
            self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "label": self.labels[idx],
            "path": row["image_path"],
        }


def build_label_map(species_set):
    """Map species names to integer indices.

    Returns:
        label_map: dict {species_name: index}
        species_list: sorted list of species names (index -> name)
    """
    sorted_species = sorted(species_set)
    label_map = {sp: i for i, sp in enumerate(sorted_species)}
    return label_map, sorted_species
