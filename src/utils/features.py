import torch
import numpy as np
from tqdm import tqdm


def extract_image_features(clip_wrapper, dataloader, desc="Extracting features"):
    """Extract normalized CLIP image features from a dataloader.

    Args:
        clip_wrapper: CLIPWrapper instance
        dataloader: DataLoader yielding dicts with "pixel_values", "path", "label"
        desc: progress bar description

    Returns:
        features: (N, D) numpy array of normalized features
        metadata: list of dicts with "path" and "label" keys
    """
    features = []
    metadata = []
    clip_wrapper.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            px = batch["pixel_values"].to(clip_wrapper.device)
            feats = clip_wrapper.encode_images(px).cpu().numpy()
            features.append(feats)
            for p, l in zip(batch["path"], batch["label"]):
                if isinstance(l, torch.Tensor):
                    label_val = l.item()
                elif isinstance(l, str):
                    label_val = l
                else:
                    label_val = int(l)
                metadata.append({"path": p, "label": label_val})
    return np.vstack(features), metadata
