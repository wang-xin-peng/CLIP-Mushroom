# CLIP-Mushroom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete fungi fine-grained recognition and multi-modal retrieval system using CLIP, with zero-shot classification, Linear Probe fine-tuning, FAISS-based image/text retrieval, explainability heatmaps, and a Gradio demo.

**Architecture:** The system loads CLIP ViT-B/16, freezes the image encoder, and performs (1) zero-shot classification via text prompts on all classes, (2) linear probe fine-tuning on seen classes, (3) feature extraction + FAISS indexing for image/text-to-image retrieval, (4) attention-based heatmap visualization, and (5) a 3-tab Gradio web interface.

**Tech Stack:** PyTorch, transformers (HuggingFace CLIP), FAISS-cpu, scikit-learn, Gradio, matplotlib, seaborn, pandas

---

### Task 1: Data loading and seen/unseen split

**Files:**
- Create: `src/data/dataset.py`
- Create: `src/data/split.py`

- [ ] **Step 1: Create `src/data/split.py` — load metadata and split species into seen/unseen**

```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path("data/FungiTastic/FungiTastic-Mini")
META_ROOT = Path("data/FungiTastic/metadata/FungiTastic-Mini")

def get_metadata_splits():
    """Load train/val/test metadata and return DataFrames with image paths."""
    splits = {}
    for name in ["Train", "Test", "Val"]:
        df = pd.read_csv(META_ROOT / f"FungiTastic-Mini-{name}.csv")
        df["image_path"] = df["filename"].apply(
            lambda x: str(DATA_ROOT / name.lower() / "300p" / x) if pd.notna(x) else ""
        )
        df["label"] = df["scientificName"].fillna("unknown")
        # Keep only rows with a valid filename
        df = df[df["filename"].notna() & (df["filename"] != "")]
        # Drop duplicates by filename (each image is one sample)
        df = df.drop_duplicates(subset=["filename"])
        splits[name.lower()] = df
    return splits["train"], splits["val"], splits["test"]

def split_species(train_df, unseen_ratio=0.2, random_seed=42):
    """Split species into seen (for training) and unseen (for zero-shot test)."""
    rng = np.random.RandomState(random_seed)
    all_species = sorted(train_df["label"].unique())
    n_unseen = max(1, int(len(all_species) * unseen_ratio))
    unseen_species = set(rng.choice(all_species, n_unseen, replace=False))
    seen_species = set(all_species) - unseen_species
    return seen_species, unseen_species
```

- [ ] **Step 2: Quick verification of split**

Run:
```bash
eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)" && conda activate E:/conda_envs/clip-mushroom && cd E:/ComputerVision/CLIP-Mushroom && python -c "
from src.data.split import get_metadata_splits, split_species
train, val, test = get_metadata_splits()
print(f'Train: {len(train)} images, Val: {len(val)} images, Test: {len(test)} images')
seen, unseen = split_species(train)
print(f'Seen species: {len(seen)}, Unseen species: {len(unseen)}')
"
```
Expected: Train/val/test counts printed, roughly 80% seen / 20% unseen species.

- [ ] **Step 3: Create `src/data/dataset.py` — PyTorch Dataset for CLIP**

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor

class FungiDataset(Dataset):
    def __init__(self, df, label_map=None, processor=None, split="train"):
        self.df = df.reset_index(drop=True)
        self.processor = processor or CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        if label_map is not None:
            self.labels = df["label"].map(label_map).values
        else:
            self.labels = df["label"].values
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = self.labels[idx]
        return {"pixel_values": pixel_values, "label": label, "path": row["image_path"]}

def build_label_map(species_set):
    """Map species names to integer indices."""
    sorted_species = sorted(species_set)
    return {sp: i for i, sp in enumerate(sorted_species)}, sorted_species
```

- [ ] **Step 4: Commit**

```bash
cd E:/ComputerVision/CLIP-Mushroom
git add src/data/dataset.py src/data/split.py
git commit -m "feat: add data loading and seen/unseen species split"
```

---

### Task 2: Zero-shot CLIP classification

**Files:**
- Create: `src/utils/prompts.py`
- Create: `src/models/clip_wrapper.py`

- [ ] **Step 1: Create `src/utils/prompts.py` — prompt templates**

```python
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a macro shot of a {} fungus",
    "the mushroom {}",
    "{} growing in the wild",
    "a close-up photograph of {}",
]

def get_prompts_for_classes(class_names, template="a photo of {}"):
    """Generate prompt texts for a list of class names using the given template."""
    return [template.format(name) for name in class_names]

def get_all_prompt_variants(class_names):
    """Return dict mapping template name -> list of prompt texts."""
    return {
        f"template_{i}": get_prompts_for_classes(class_names, t)
        for i, t in enumerate(PROMPT_TEMPLATES)
    }
```

- [ ] **Step 2: Create `src/models/clip_wrapper.py` — CLIP model wrapper**

```python
import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPWrapper:
    def __init__(self, model_name="openai/clip-vit-base-patch16", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_images(self, pixel_values):
        """Extract image features. Input: (B, C, H, W) tensor. Output: (B, D) normalized features."""
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values.to(self.device))
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, text_list):
        """Extract text features. Input: list of strings. Output: (N, D) normalized features."""
        with torch.no_grad():
            inputs = self.processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def zero_shot_predict(self, pixel_values, class_names, template="a photo of {}"):
        """Predict class probabilities for images given class names + prompt template."""
        prompts = [template.format(name) for name in class_names]
        text_features = self.encode_text(prompts)  # (C, D)
        image_features = self.encode_images(pixel_values)  # (B, D)
        logits = image_features @ text_features.T * self.model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1)
        return probs, logits
```

- [ ] **Step 3: Verify zero-shot works minimally**

Run:
```bash
eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)" && conda activate E:/conda_envs/clip-mushroom && cd E:/ComputerVision/CLIP-Mushroom && python -c "
from src.models.clip_wrapper import CLIPWrapper
import torch
model = CLIPWrapper()
print(f'Device: {model.device}')
# Test with dummy input
dummy = torch.randn(1, 3, 224, 224)
probs, _ = model.zero_shot_predict(dummy, ['Agaricus bisporus', 'Amanita muscaria'])
print(f'Probabilities shape: {probs.shape}')
print('Zero-shot OK')
"
```
Expected: Device shown, probabilities tensor of shape (1, 2).

- [ ] **Step 4: Commit**

```bash
git add src/utils/prompts.py src/models/clip_wrapper.py
git commit -m "feat: add CLIP model wrapper and prompt templates for zero-shot"
```

---

### Task 3: Zero-shot evaluation script

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `src/main.py` (zero-shot section)

- [ ] **Step 1: Create `src/evaluation/metrics.py`**

```python
import torch
import numpy as np
import pandas as pd

def accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy. output: (N, C) logits/probs, target: (N,) class indices."""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        res.append(correct_k)
    return res  # returns [top1_correct_count, top5_correct_count]

def evaluate_zero_shot(model, dataloader, class_names, template):
    """Run zero-shot evaluation on a dataloader. Returns accuracy dict and all predictions."""
    model.model.eval()
    all_preds = []
    all_labels = []
    top1_total = 0
    top5_total = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            labels = batch["label"]
            probs, logits = model.zero_shot_predict(pixel_values, class_names, template)
            batch_top1, batch_top5 = accuracy(logits.cpu(), labels, topk=(1, 5))
            top1_total += batch_top1
            top5_total += batch_top5
            total += labels.size(0)
            all_preds.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "top1": top1_total / total * 100,
        "top5": top5_total / total * 100,
        "total": total,
        "logits": all_logits,
        "labels": all_labels,
    }
```

- [ ] **Step 2: Create test runner in `src/main.py` (zero-shot part)**

```python
import torch
from torch.utils.data import DataLoader
from src.models.clip_wrapper import CLIPWrapper
from src.data.dataset import FungiDataset, build_label_map
from src.data.split import get_metadata_splits, split_species
from src.utils.prompts import PROMPT_TEMPLATES
from src.evaluation.metrics import evaluate_zero_shot
import pandas as pd
import json

def run_zero_shot_test():
    print("Loading data...")
    train_df, val_df, test_df = get_metadata_splits()
    seen_species, unseen_species = split_species(train_df)
    all_species = sorted(seen_species | unseen_species)
    _, species_list = build_label_map(all_species)
    
    clip = CLIPWrapper()
    processor = clip.processor
    _, all_species_names = build_label_map(all_species)

    # Build test dataloader with numeric labels
    label_map, _ = build_label_map(all_species)
    test_dataset = FungiDataset(test_df, label_map=label_map, processor=processor, split="test")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    results = []
    for i, template in enumerate(PROMPT_TEMPLATES):
        print(f"Evaluating template {i}: '{template}'")
        result = evaluate_zero_shot(clip, test_loader, all_species_names, template)
        results.append({
            "template": template,
            "top1": result["top1"],
            "top5": result["top5"],
        })
        print(f"  Top-1: {result['top1']:.2f}%, Top-5: {result['top5']:.2f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/reports/zero_shot_results.csv", index=False)
    print("\nBest template:")
    best = results_df.loc[results_df["top1"].idxmax()]
    print(f"  {best['template']} -> Top-1: {best['top1']:.2f}%, Top-5: {best['top5']:.2f}%")

if __name__ == "__main__":
    run_zero_shot_test()
```

- [ ] **Step 3: Run zero-shot evaluation**

Run:
```bash
eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)" && conda activate E:/conda_envs/clip-mushroom && cd E:/ComputerVision/CLIP-Mushroom && python src/main.py
```
Expected: Zero-shot evaluation results for each template printed, CSV saved.

- [ ] **Step 4: Commit**

```bash
git add src/evaluation/metrics.py src/main.py
git commit -m "feat: add zero-shot evaluation with multiple prompt templates"
```

---

### Task 4: Linear Probe training

**Files:**
- Create: `src/models/linear_probe.py`

- [ ] **Step 1: Create `src/models/linear_probe.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

class LinearProbe(nn.Module):
    """Linear classifier on top of frozen CLIP image features."""
    def __init__(self, feature_dim=512, num_classes=1000):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_linear_probe(clip_wrapper, train_dataset, val_dataset, num_classes,
                       epochs=20, lr=1e-3, batch_size=64, device=None):
    """Train linear probe on CLIP features. Returns trained model and history."""
    device = device or clip_wrapper.device
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Pre-extract features for speed
    print("Pre-extracting training features...")
    train_features, train_labels = [], []
    clip_wrapper.model.eval()
    with torch.no_grad():
        for batch in train_loader:
            px = batch["pixel_values"].to(device)
            feats = clip_wrapper.encode_images(px).cpu()
            train_features.append(feats)
            train_labels.append(batch["label"])
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    print("Pre-extracting validation features...")
    val_features, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            px = batch["pixel_values"].to(device)
            feats = clip_wrapper.encode_images(px).cpu()
            val_features.append(feats)
            val_labels.append(batch["label"])
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    model = LinearProbe(feature_dim=train_features.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_features.size(0))
        train_features_shuffled = train_features[perm]
        train_labels_shuffled = train_labels[perm]
        
        epoch_loss = 0.0
        for i in range(0, len(train_features), batch_size):
            batch_feats = train_features_shuffled[i:i+batch_size].to(device)
            batch_labels = train_labels_shuffled[i:i+batch_size].to(device)
            
            outputs = model(batch_feats)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(val_features.to(device))
            _, predicted = torch.max(outputs, 1)
            total += val_labels.size(0)
            correct += (predicted.cpu() == val_labels).sum().item()
        val_acc = 100.0 * correct / total
        
        history["train_loss"].append(epoch_loss / max(1, len(train_features) // batch_size))
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {history['train_loss'][-1]:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/linear_probe_best.pt")

    model.load_state_dict(torch.load("checkpoints/linear_probe_best.pt"))
    return model, history
```

- [ ] **Step 2: Add Linear Probe training to `src/main.py`**

```python
def run_linear_probe():
    from src.data.split import get_metadata_splits, split_species
    from src.data.dataset import FungiDataset, build_label_map
    from src.models.clip_wrapper import CLIPWrapper
    from src.models.linear_probe import train_linear_probe
    
    train_df, val_df, test_df = get_metadata_splits()
    seen_species, unseen_species = split_species(train_df)
    seen_species_list = sorted(seen_species)
    print(f"Training on {len(seen_species_list)} seen species")
    
    clip = CLIPWrapper()
    processor = clip.processor
    
    label_map, _ = build_label_map(seen_species_list)
    train_dataset = FungiDataset(train_df[train_df["label"].isin(seen_species_list)],
                                  label_map=label_map, processor=processor, split="train")
    val_dataset = FungiDataset(val_df[val_df["label"].isin(seen_species_list)],
                                label_map=label_map, processor=processor, split="val")
    
    model, history = train_linear_probe(clip, train_dataset, val_dataset, len(seen_species_list))
    
    # Save training curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"])
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"])
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig("outputs/figures/linear_probe_training.png")
    plt.close()
    print("Training complete. Best validation accuracy: {:.2f}%".format(max(history["val_acc"])))
```

- [ ] **Step 3: Run Linear Probe training**

Run:
```bash
eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)" && conda activate E:/conda_envs/clip-mushroom && cd E:/ComputerVision/CLIP-Mushroom && python -c "
from src.main import run_linear_probe
run_linear_probe()
"
```
Expected: Training logs showing decreasing loss and increasing val accuracy, best model saved.

- [ ] **Step 4: Evaluate Linear Probe on test set (seen classes)**

```python
def evaluate_linear_probe():
    import torch
    from torch.utils.data import DataLoader
    from src.models.linear_probe import LinearProbe
    from src.models.clip_wrapper import CLIPWrapper
    from src.data.dataset import FungiDataset, build_label_map
    from src.data.split import get_metadata_splits, split_species
    from src.evaluation.metrics import accuracy
    
    train_df, val_df, test_df = get_metadata_splits()
    seen_species, unseen_species = split_species(train_df)
    seen_species_list = sorted(seen_species)
    
    clip = CLIPWrapper()
    label_map, _ = build_label_map(seen_species_list)
    test_seen = test_df[test_df["label"].isin(seen_species_list)]
    test_dataset = FungiDataset(test_seen, label_map=label_map, processor=clip.processor, split="test")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    num_classes = len(seen_species_list)
    model = LinearProbe(num_classes=num_classes).to(clip.device)
    model.load_state_dict(torch.load("checkpoints/linear_probe_best.pt"))
    model.eval()
    
    correct_1, correct_5, total = 0, 0, 0
    clip.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            px = batch["pixel_values"].to(clip.device)
            feats = clip.encode_images(px)
            outputs = model(feats)
            c1, c5 = accuracy(outputs.cpu(), batch["label"], topk=(1, 5))
            correct_1 += c1
            correct_5 += c5
            total += batch["label"].size(0)
    
    print(f"Linear Probe - Top-1: {100*correct_1/total:.2f}%, Top-5: {100*correct_5/total:.2f}%")
```

- [ ] **Step 5: Commit**

```bash
git add src/models/linear_probe.py src/main.py
git commit -m "feat: add Linear Probe training and evaluation"
```

---

### Task 5: FAISS retrieval system

**Files:**
- Create: `src/utils/features.py`
- Create: `src/utils/faiss_index.py`

- [ ] **Step 1: Create `src/utils/features.py`**

```python
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

def extract_image_features(clip_wrapper, dataloader):
    """Extract normalized image features from a dataloader. Returns (N, D) numpy array + metadata."""
    features = []
    metadata = []
    clip_wrapper.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            px = batch["pixel_values"].to(clip_wrapper.device)
            feats = clip_wrapper.encode_images(px).cpu().numpy()
            features.append(feats)
            metadata.extend([
                {"path": p, "label": l}
                for p, l in zip(batch["path"], batch["label"])
            ])
    return np.vstack(features), metadata
```

- [ ] **Step 2: Create `src/utils/faiss_index.py`**

```python
import faiss
import numpy as np
import torch

class FaissIndex:
    def __init__(self, dim=512, metric="cosine"):
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
        else:
            self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # list of dicts

    def add(self, features, metadata):
        """Add (N, D) numpy features and corresponding metadata list."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        features = features.astype(np.float32)
        self.index.add(features)
        self.metadata.extend(metadata)

    def search(self, query_vector, k=10):
        """Search for top-k nearest neighbors. query_vector: (D,) or (1, D). Returns indices, scores, metadata."""
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.cpu().numpy()
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({**self.metadata[idx], "score": float(score)})
        return results

    def save(self, path):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, str(path))
        import pickle
        with open(str(path) + ".meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path):
        """Load FAISS index and metadata."""
        self.index = faiss.read_index(str(path))
        import pickle
        with open(str(path) + ".meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)
```

- [ ] **Step 3: Build the index**

```python
def build_retrieval_index():
    import pickle
    from src.data.split import get_metadata_splits
    from src.data.dataset import FungiDataset
    from src.models.clip_wrapper import CLIPWrapper
    from src.utils.features import extract_image_features
    from src.utils.faiss_index import FaissIndex
    from torch.utils.data import DataLoader
    
    train_df, val_df, test_df = get_metadata_splits()
    clip = CLIPWrapper()
    
    # Build full train dataset (no label filtering)
    all_labels = sorted(train_df["label"].unique())
    label_map = {l: i for i, l in enumerate(all_labels)}
    
    train_dataset = FungiDataset(train_df, label_map=label_map, processor=clip.processor, split="train")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    features, metadata = extract_image_features(clip, train_loader)
    
    index = FaissIndex(dim=features.shape[1])
    index.add(features, metadata)
    index.save("outputs/features/faiss_index.bin")
    print(f"Index built: {len(metadata)} images, feature dim {features.shape[1]}")
    
    # Also save label map for retrieval display
    rev_label_map = {v: k for k, v in label_map.items()}
    with open("outputs/features/label_map.pkl", "wb") as f:
        pickle.dump({"label_map": label_map, "rev_label_map": rev_label_map}, f)
```

- [ ] **Step 4: Verify retrieval**

Run:
```bash
eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)" && conda activate E:/conda_envs/clip-mushroom && cd E:/ComputerVision/CLIP-Mushroom && python -c "
from src.main import build_retrieval_index
build_retrieval_index()
"
```
Expected: Feature extraction progress bar, "Index built: X images" message.

- [ ] **Step 5: Implement text-to-image retrieval**

```python
def text_to_image_search(clip_wrapper, faiss_index, query_text, k=10):
    """Search images by text description."""
    text_feat = clip_wrapper.encode_text([query_text])
    results = faiss_index.search(text_feat, k=k)
    return results

def image_to_image_search(clip_wrapper, faiss_index, pixel_values, k=10):
    """Search images by image query."""
    img_feat = clip_wrapper.encode_images(pixel_values)
    results = faiss_index.search(img_feat, k=k)
    return results
```

- [ ] **Step 6: Commit**

```bash
git add src/utils/features.py src/utils/faiss_index.py src/main.py
git commit -m "feat: add FAISS retrieval system with image and text search"
```

---

### Task 6: Evaluation and comparison reports

**Files:**
- Modify: `src/evaluation/metrics.py`
- Create: `src/evaluation/report.py`

- [ ] **Step 1: Add report generation to `src/evaluation/report.py`**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_zero_shot_comparison(csv_path="outputs/reports/zero_shot_results.csv",
                               save_path="outputs/figures/zero_shot_comparison.png"):
    """Bar chart comparing prompt templates."""
    df = pd.read_csv(csv_path)
    x = np.arange(len(df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, df["top1"], width, label="Top-1")
    bars2 = ax.bar(x + width/2, df["top5"], width, label="Top-5")
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot CLIP Performance by Prompt Template")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(len(df))])
    ax.legend()
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Chart saved to {save_path}")

def plot_comparison_table(zero_shot_csv="outputs/reports/zero_shot_results.csv",
                           lp_report_path="outputs/reports/linear_probe_results.csv",
                           save_path="outputs/figures/comparison_table.png"):
    """Generate a comparison table between zero-shot and linear probe."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ["Method", "Seen Top-1", "Seen Top-5", "Unseen Top-1", "Unseen Top-5"],
    ]
    
    # Add zero-shot best
    zs_df = pd.read_csv(zero_shot_csv)
    best_zs = zs_df.loc[zs_df["top1"].idxmax()]
    data.append(["Zero-shot (best)", f"{best_zs['top1']:.2f}%", f"{best_zs['top5']:.2f}%", "-", "-"])
    
    # Add linear probe if exists
    try:
        lp_df = pd.read_csv(lp_report_path)
        data.append(["Linear Probe", f"{lp_df['seen_top1'][0]:.2f}%", f"{lp_df['seen_top5'][0]:.2f}%", "-", "-"])
    except:
        data.append(["Linear Probe", "N/A", "N/A", "-", "-"])
    
    table = ax.table(cellText=data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison table saved to {save_path}")
```

- [ ] **Step 2: Generate comparison report**

Run:
```bash
eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)" && conda activate E:/conda_envs/clip-mushroom && cd E:/ComputerVision/CLIP-Mushroom && python -c "
from src.evaluation.report import plot_zero_shot_comparison, plot_comparison_table
plot_zero_shot_comparison()
plot_comparison_table()
"
```
Expected: Charts saved to outputs/figures/.

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/report.py
git commit -m "feat: add evaluation report generation with charts"
```

---

### Task 7: Grad-CAM explainability

**Files:**
- Create: `src/utils/gradcam.py`

- [ ] **Step 1: Create Grad-CAM for CLIP ViT**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class CLIPGradCAM:
    """Grad-CAM for CLIP ViT using attention map from the last layer."""
    def __init__(self, clip_wrapper):
        self.model = clip_wrapper.model
        self.device = clip_wrapper.device
        self.processor = clip_wrapper.processor
        
        # Register hook on the vision encoder's final layer
        self.attention_map = None
        self.vision_model = self.model.vision_model
        self.encoder = self.vision_model.encoder
        self.last_layer = self.encoder.layers[-1]
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            # output[0] is the hidden states, output[1] is the attention weights
            # For CLIP ViT, we need self-attention from the last layer
            self.attention_map = output[1]  # (batch, heads, seq_len, seq_len)
        self.last_layer.self_attn.register_forward_hook(hook)

    def generate_heatmap(self, pixel_values, class_idx=None):
        """Generate Grad-CAM heatmap. class_idx=None uses the predicted class."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values.to(self.device),
                                return_dict=True)
            
        # Get the attention map from the [CLS] token to all patch tokens
        # attention_map shape: (batch, heads, seq_len, seq_len)
        attn = self.attention_map.cpu()
        # Average over heads
        attn = attn.mean(dim=1)  # (batch, seq_len, seq_len)
        # Get CLS attention to patches: attn[0, 0, 1:] 
        cls_attn = attn[0, 0, 1:]  # exclude CLS self-attention
        num_patches = int(np.sqrt(cls_attn.shape[0]))
        
        # Reshape to 2D grid
        heatmap = cls_attn.reshape(num_patches, num_patches).numpy()
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap

    def visualize(self, pixel_values, image_path, save_path=None):
        """Generate and overlay heatmap on the original image."""
        heatmap = self.generate_heatmap(pixel_values)
        
        # Load original image
        img = Image.open(image_path).convert("RGB")
        
        # Resize heatmap to match original image
        heatmap_resized = np.array(Image.fromarray(np.uint8(heatmap * 255)).resize(img.size, Image.BICUBIC))
        
        # Overlay
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(img)
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        ax2.imshow(heatmap_resized, cmap="jet", alpha=0.7)
        ax2.set_title("Attention Heatmap")
        ax2.axis("off")
        
        ax3.imshow(img)
        ax3.imshow(heatmap_resized, cmap="jet", alpha=0.5)
        ax3.set_title("Overlay")
        ax3.axis("off")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
```

- [ ] **Step 2: Test Grad-CAM on sample images**

```python
def test_gradcam():
    from src.models.clip_wrapper import CLIPWrapper
    from src.utils.gradcam import CLIPGradCAM
    from src.data.split import get_metadata_splits
    from pathlib import Path
    import os
    
    train_df, _, _ = get_metadata_splits()
    clip = CLIPWrapper()
    gradcam = CLIPGradCAM(clip)
    processor = clip.processor
    
    os.makedirs("outputs/figures/gradcam", exist_ok=True)
    samples = train_df.head(5)
    
    for idx, row in samples.iterrows():
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        save_path = f"outputs/figures/gradcam/gradcam_{Path(img_path).stem}.png"
        gradcam.visualize(inputs["pixel_values"], img_path, save_path)
        print(f"Saved: {save_path}")
```

- [ ] **Step 3: Run Grad-CAM**

Run:
```bash
eval "$(E:/anaconda/Scripts/conda.exe shell.bash hook)" && conda activate E:/conda_envs/clip-mushroom && cd E:/ComputerVision/CLIP-Mushroom && python -c "
from src.main import test_gradcam
test_gradcam()
"
```
Expected: 5 heatmap images saved to outputs/figures/gradcam/.

- [ ] **Step 4: Commit**

```bash
git add src/utils/gradcam.py src/main.py
git commit -m "feat: add Grad-CAM attention visualization for CLIP ViT"
```

---

### Task 8: Gradio web interface

**Files:**
- Create: `src/gradio_app/app.py`

- [ ] **Step 1: Create Gradio app**

```python
import gradio as gr
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import pickle

from src.models.clip_wrapper import CLIPWrapper
from src.utils.faiss_index import FaissIndex
from src.utils.gradcam import CLIPGradCAM
from src.utils.prompts import PROMPT_TEMPLATES
from src.data.split import get_metadata_splits, split_species

# Global state (loaded once)
clip = None
faiss_index = None
gradcam = None
all_species = None
species_list = None
rev_label_map = None
label_map = None

def load_models():
    global clip, faiss_index, gradcam, all_species, species_list, rev_label_map, label_map
    if clip is None:
        clip = CLIPWrapper()
        # Load FAISS index
        faiss_index = FaissIndex()
        faiss_index.load("outputs/features/faiss_index.bin")
        with open("outputs/features/label_map.pkl", "rb") as f:
            data = pickle.load(f)
            rev_label_map = data["rev_label_map"]
            label_map = data["label_map"]
        gradcam = CLIPGradCAM(clip)
        # Species for zero-shot
        train_df, _, _ = get_metadata_splits()
        seen, unseen = split_species(train_df)
        all_species = sorted(seen | unseen)
        species_list = all_species

# Tab 1: Zero-shot recognition
def recognize_image(img):
    load_models()
    processor = clip.processor
    image = Image.fromarray(img).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Zero-shot prediction
    probs, _ = clip.zero_shot_predict(inputs["pixel_values"], species_list, PROMPT_TEMPLATES[0])
    top5_probs, top5_indices = torch.topk(probs[0], 5)
    
    results = []
    for p, idx in zip(top5_probs, top5_indices):
        results.append({species_list[idx]: float(p)})
    
    # Generate heatmap
    gradcam_result = gradcam.visualize(inputs["pixel_values"], None)
    buf = io.BytesIO()
    gradcam_result.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(gradcam_result)
    buf.seek(0)
    heatmap_img = Image.open(buf)
    
    return results, heatmap_img

# Tab 2: Image search
def search_by_image(img, k=10):
    load_models()
    processor = clip.processor
    image = Image.fromarray(img).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    query_feat = clip.encode_images(inputs["pixel_values"])
    results = faiss_index.search(query_feat, k=k)
    
    # Return gallery of results
    result_images = []
    for r in results:
        result_images.append((r["path"], f"{rev_label_map.get(r['label'], 'unknown')} (score: {r['score']:.3f})"))
    return result_images

# Tab 3: Text search
def search_by_text(query, k=10):
    load_models()
    text_feat = clip.encode_text([query])
    results = faiss_index.search(text_feat, k=k)
    
    result_images = []
    for r in results:
        result_images.append((r["path"], f"{rev_label_map.get(r['label'], 'unknown')} (score: {r['score']:.3f})"))
    return result_images

def create_app():
    with gr.Blocks(title="CLIP-Mushroom: Fungi Recognition & Retrieval", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🍄 CLIP-Mushroom: 真菌细粒度识别与多模态检索系统")
        
        with gr.Tab("零样本识别 (Zero-shot)"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="上传真菌图片")
                    recognize_btn = gr.Button("识别")
                with gr.Column():
                    output_preds = gr.Label(label="预测结果 (Top-5)")
                    output_heatmap = gr.Image(label="注意力热力图")
            recognize_btn.click(fn=recognize_image, inputs=input_img, outputs=[output_preds, output_heatmap])
        
        with gr.Tab("以图搜图 (Image Search)"):
            with gr.Row():
                with gr.Column():
                    search_img = gr.Image(label="上传查询图片")
                    img_k = gr.Slider(5, 50, value=10, step=5, label="返回数量 K")
                    img_search_btn = gr.Button("搜索")
                with gr.Column():
                    img_gallery = gr.Gallery(label="检索结果", columns=5, height="auto")
            img_search_btn.click(fn=search_by_image, inputs=[search_img, img_k], outputs=img_gallery)
        
        with gr.Tab("文本搜图 (Text Search)"):
            with gr.Row():
                with gr.Column():
                    text_query = gr.Textbox(label="输入文字描述", placeholder="e.g., red mushroom with white spots")
                    text_k = gr.Slider(5, 50, value=10, step=5, label="返回数量 K")
                    text_search_btn = gr.Button("搜索")
                with gr.Column():
                    text_gallery = gr.Gallery(label="检索结果", columns=5, height="auto")
            text_search_btn.click(fn=search_by_text, inputs=[text_query, text_k], outputs=text_gallery)
    
    return app
```

- [ ] **Step 2: Add app launch entry**

```python
# In src/main.py, add:
def launch_gradio():
    from src.gradio_app.app import create_app
    app = create_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
```

- [ ] **Step 3: Commit**

```bash
git add src/gradio_app/app.py src/main.py
git commit -m "feat: add Gradio web interface with 3 tabs"
```

---

### Task 9: Full pipeline integration and final main.py

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Write the unified `src/main.py` with CLI argument support**

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="CLIP-Mushroom: Fungi Recognition & Retrieval")
    parser.add_argument("--zero-shot", action="store_true", help="Run zero-shot evaluation")
    parser.add_argument("--linear-probe", action="store_true", help="Train and evaluate Linear Probe")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS retrieval index")
    parser.add_argument("--eval", action="store_true", help="Generate evaluation reports")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmaps")
    parser.add_argument("--app", action="store_true", help="Launch Gradio web app")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    
    args = parser.parse_args()
    
    if args.all or args.zero_shot:
        run_zero_shot_test()
    if args.all or args.linear_probe:
        run_linear_probe()
    if args.all or args.build_index:
        build_retrieval_index()
    if args.all or args.eval:
        from src.evaluation.report import plot_zero_shot_comparison
        plot_zero_shot_comparison()
    if args.all or args.gradcam:
        test_gradcam()
    if args.all or args.app:
        launch_gradio()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: integrate full pipeline with CLI arguments"
```

---
