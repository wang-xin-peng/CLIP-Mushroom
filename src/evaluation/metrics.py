import torch
import numpy as np
import pandas as pd


def accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy.

    Args:
        output: (N, C) logits or probabilities
        target: (N,) class indices
        topk: tuple of k values to evaluate

    Returns:
        list of [top1_correct_count, top5_correct_count, ...]
    """
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def evaluate_zero_shot(model, dataloader, class_names, template):
    """Run zero-shot evaluation on a dataloader.

    Returns:
        dict with top1, top5 accuracy (%), total count, logits, and labels
    """
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
            # labels are already class indices
            label_indices = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
            probs, logits = model.zero_shot_predict(pixel_values, class_names, template)
            batch_top1, batch_top5 = accuracy(logits.cpu(), label_indices, topk=(1, 5))
            top1_total += batch_top1
            top5_total += batch_top5
            total += label_indices.size(0)
            all_preds.append(logits.cpu())
            all_labels.append(label_indices)

    all_logits = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "top1": top1_total / total * 100,
        "top5": top5_total / total * 100,
        "total": total,
        "logits": all_logits,
        "labels": all_labels,
    }


def evaluate_linear_probe(clip_wrapper, linear_model, dataloader, topk=(1, 5)):
    """Evaluate a trained Linear Probe on a dataloader.

    Returns:
        dict with top1, top5 accuracy (%) and total count
    """
    clip_wrapper.model.eval()
    linear_model.eval()
    correct_1, correct_5, total = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            px = batch["pixel_values"].to(clip_wrapper.device)
            labels = batch["label"]
            label_indices = torch.tensor(labels).to(clip_wrapper.device) if not isinstance(labels, torch.Tensor) else labels.to(clip_wrapper.device)
            feats = clip_wrapper.encode_images(px)
            outputs = linear_model(feats)
            c1, c5 = accuracy(outputs.cpu(), label_indices.cpu(), topk=topk)
            correct_1 += c1
            correct_5 += c5
            total += label_indices.size(0)

    return {
        "top1": correct_1 / total * 100,
        "top5": correct_5 / total * 100,
        "total": total,
    }
