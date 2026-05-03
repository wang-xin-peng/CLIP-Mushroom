"""CLIP-Mushroom: 真菌细粒度识别与多模态检索系统

Usage:
    python src/main.py --zero-shot     # Run zero-shot evaluation
    python src/main.py --linear-probe   # Train and evaluate Linear Probe
    python src/main.py --build-index    # Build FAISS retrieval index
    python src/main.py --eval           # Generate evaluation reports
    python src/main.py --gradcam        # Generate Grad-CAM heatmaps
    python src/main.py --app            # Launch Gradio web app
    python src/main.py --all            # Run full pipeline
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Task 3: Zero-shot Evaluation
# ──────────────────────────────────────────────

def run_zero_shot_test():
    from src.data.dataset import FungiDataset, build_label_map
    from src.data.split import get_metadata_splits, split_species
    from src.models.clip_wrapper import CLIPWrapper
    from src.utils.prompts import PROMPT_TEMPLATES
    from src.evaluation.metrics import evaluate_zero_shot

    print("=" * 60)
    print("Task 3: Zero-shot CLIP Classification")
    print("=" * 60)

    print("\nLoading data...")
    train_df, val_df, test_df = get_metadata_splits()
    seen_species, unseen_species = split_species(train_df)
    all_species = sorted(seen_species | unseen_species)
    print(f"Total species: {len(all_species)} (seen: {len(seen_species)}, unseen: {len(unseen_species)})")
    print(f"Test set: {len(test_df)} images")

    print("\nLoading CLIP model...")
    clip = CLIPWrapper()
    processor = clip.processor

    # Build label map for all species (for consistent indexing)
    label_map, species_list = build_label_map(all_species)

    # Test dataloader
    test_dataset = FungiDataset(test_df, label_map=label_map, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Test batches: {len(test_loader)}")

    results = []
    for i, template in enumerate(PROMPT_TEMPLATES):
        print(f"\nEvaluating template {i}: '{template}'")
        result = evaluate_zero_shot(clip, test_loader, species_list, template)
        results.append({
            "template": template,
            "top1": result["top1"],
            "top5": result["top5"],
            "total": result["total"],
        })
        print(f"  Top-1: {result['top1']:.2f}%, Top-5: {result['top5']:.2f}%")

    results_df = pd.DataFrame(results)
    os.makedirs("outputs/reports", exist_ok=True)
    results_df.to_csv("outputs/reports/zero_shot_results.csv", index=False)
    print("\nResults saved to outputs/reports/zero_shot_results.csv")

    best = results_df.loc[results_df["top1"].idxmax()]
    print(f"\nBest template: \"{best['template']}\"")
    print(f"  Top-1: {best['top1']:.2f}%, Top-5: {best['top5']:.2f}%")
    print()


# ──────────────────────────────────────────────
# Task 4: Linear Probe Training
# ──────────────────────────────────────────────

def run_linear_probe():
    from src.data.dataset import FungiDataset, build_label_map
    from src.data.split import get_metadata_splits, split_species
    from src.models.clip_wrapper import CLIPWrapper
    from src.models.linear_probe import train_linear_probe
    from src.evaluation.metrics import evaluate_linear_probe

    print("=" * 60)
    print("Task 4: Linear Probe Training")
    print("=" * 60)

    print("\nLoading data...")
    train_df, val_df, test_df = get_metadata_splits()
    seen_species, unseen_species = split_species(train_df)
    seen_list = sorted(seen_species)
    print(f"Training on {len(seen_list)} seen species")
    print(f"Train samples (seen only): {len(train_df[train_df['label'].isin(seen_list)])}")

    print("\nLoading CLIP model...")
    clip = CLIPWrapper()
    processor = clip.processor

    # Build label map for seen species only
    label_map, _ = build_label_map(seen_list)

    # Filter datasets to seen species
    train_seen = train_df[train_df["label"].isin(seen_list)]
    val_seen = val_df[val_df["label"].isin(seen_list)]
    test_seen = test_df[test_df["label"].isin(seen_list)]

    train_dataset = FungiDataset(train_seen, label_map=label_map, processor=processor)
    val_dataset = FungiDataset(val_seen, label_map=label_map, processor=processor)
    test_dataset = FungiDataset(test_seen, label_map=label_map, processor=processor)

    # Train
    model, history = train_linear_probe(
        clip, train_dataset, val_dataset,
        num_classes=len(seen_list),
        epochs=20, lr=1e-3, batch_size=64,
    )

    # Evaluate on test set (seen classes only)
    print("\nEvaluating on test set (seen classes)...")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_results = evaluate_linear_probe(clip, model, test_loader)
    print(f"Test Top-1: {test_results['top1']:.2f}%, Top-5: {test_results['top5']:.2f}%")

    # Save results
    os.makedirs("outputs/reports", exist_ok=True)
    pd.DataFrame([{
        "seen_top1": test_results["top1"],
        "seen_top5": test_results["top5"],
        "best_val_acc": max(history["val_acc"]),
    }]).to_csv("outputs/reports/linear_probe_results.csv", index=False)

    # Plot training curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("outputs/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"])
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(history["val_acc"])
    ax2.set_title("Validation Accuracy (%)")
    ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("outputs/figures/linear_probe_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nTraining curves saved to outputs/figures/linear_probe_training.png")


# ──────────────────────────────────────────────
# Task 5: Build FAISS Retrieval Index
# ──────────────────────────────────────────────

def build_retrieval_index():
    from src.data.dataset import FungiDataset
    from src.data.split import get_metadata_splits
    from src.models.clip_wrapper import CLIPWrapper
    from src.utils.features import extract_image_features
    from src.utils.faiss_index import FaissIndex

    print("=" * 60)
    print("Task 5: Build FAISS Retrieval Index")
    print("=" * 60)

    print("\nLoading data...")
    train_df, val_df, test_df = get_metadata_splits()
    print(f"Training set: {len(train_df)} images")

    print("\nLoading CLIP model...")
    clip = CLIPWrapper()
    processor = clip.processor

    # Build label map for all training labels
    all_labels = sorted(train_df["label"].unique())
    label_map = {l: i for i, l in enumerate(all_labels)}
    rev_label_map = {v: k for k, v in label_map.items()}

    train_dataset = FungiDataset(train_df, label_map=label_map, processor=processor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    print("\nExtracting features...")
    features, metadata = extract_image_features(clip, train_loader)

    print(f"\nBuilding FAISS index ({features.shape[1]} dim)...")
    index = FaissIndex(dim=features.shape[1])
    index.add(features, metadata)

    os.makedirs("outputs/features", exist_ok=True)
    index.save("outputs/features/faiss_index.bin")
    print(f"Index saved: {len(metadata)} images, feature dim {features.shape[1]}")

    # Save label maps for retrieval display
    import pickle
    with open("outputs/features/label_map.pkl", "wb") as f:
        pickle.dump({"label_map": label_map, "rev_label_map": rev_label_map}, f)
    print("Label map saved.")

    # Quick verification: image-to-image search on a test image
    print("\nVerifying retrieval with a test image...")
    sample = test_df.iloc[0]
    from PIL import Image
    img = Image.open(sample["image_path"]).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        query_feat = clip.encode_images(inputs["pixel_values"])
    results = index.search(query_feat, k=5)
    print(f"Query: {sample['label']}")
    for i, r in enumerate(results):
        print(f"  {i + 1}: {r['label']} (score: {r['score']:.4f})")
    print()


# ──────────────────────────────────────────────
# Task 7: Grad-CAM Heatmaps
# ──────────────────────────────────────────────

def test_gradcam():
    from src.models.clip_wrapper import CLIPWrapper
    from src.utils.gradcam import CLIPGradCAM
    from src.data.split import get_metadata_splits
    from PIL import Image

    print("=" * 60)
    print("Task 7: Grad-CAM Attention Heatmaps")
    print("=" * 60)

    print("\nLoading CLIP model...")
    clip = CLIPWrapper()
    gradcam = CLIPGradCAM(clip)
    processor = clip.processor

    train_df, _, _ = get_metadata_splits()
    os.makedirs("outputs/figures/gradcam", exist_ok=True)

    # Pick 5 samples from different species
    sampled = train_df.groupby("label").first().reset_index().head(5)
    print(f"Generating heatmaps for {len(sampled)} samples...")

    for idx, (_, row) in enumerate(sampled.iterrows()):
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        species_name = row["label"].replace("/", "_").replace(" ", "_")
        save_path = f"outputs/figures/gradcam/gradcam_{idx}_{species_name}.png"
        gradcam.visualize(inputs["pixel_values"], img_path, save_path)
        print(f"  [{idx + 1}/5] Saved: {save_path}")

    print("Grad-CAM complete.\n")


# ──────────────────────────────────────────────
# Task 6: Generate Evaluation Reports
# ──────────────────────────────────────────────

def generate_reports():
    from src.evaluation.report import plot_zero_shot_comparison, plot_comparison_table

    print("=" * 60)
    print("Task 6: Evaluation Reports")
    print("=" * 60)

    os.makedirs("outputs/figures", exist_ok=True)

    plot_zero_shot_comparison()
    plot_comparison_table()
    print()


# ──────────────────────────────────────────────
# Task 8: Launch Gradio App
# ──────────────────────────────────────────────

def launch_gradio():
    from src.gradio_app.app import create_app

    print("=" * 60)
    print("Task 8: Launching Gradio Web Interface")
    print("=" * 60)
    print("Open http://127.0.0.1:7860 in your browser.\n")

    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


# ──────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLIP-Mushroom: Fungi Fine-Grained Recognition & Retrieval"
    )
    parser.add_argument("--zero-shot", action="store_true", help="Run zero-shot evaluation")
    parser.add_argument("--linear-probe", action="store_true", help="Train and evaluate Linear Probe")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS retrieval index")
    parser.add_argument("--eval", action="store_true", help="Generate evaluation reports")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmaps")
    parser.add_argument("--app", action="store_true", help="Launch Gradio web app")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.all or args.zero_shot:
        run_zero_shot_test()
    if args.all or args.linear_probe:
        run_linear_probe()
    if args.all or args.build_index:
        build_retrieval_index()
    if args.all or args.eval:
        generate_reports()
    if args.all or args.gradcam:
        test_gradcam()
    if args.all or args.app:
        launch_gradio()


if __name__ == "__main__":
    main()
