import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import re
import os

from src.dataset import WrappedStoryDataset
from src.model_baseline import BaselineModel
from src.model_proposed import ProposedModel
from src.train import train
from src.evaluate import evaluate, compare
from src.utils import plot



# Utility functions

def clean_text(text):
    return re.sub(r"<[^>]+>", "", text).strip()


def preview_dataset(dataset, num_samples=2):
    print("\n========== DATASET PREVIEW ==========")

    for i in range(num_samples):
        images, text, label = dataset[i]
        print(f"\n[SAMPLE {i+1}]")
        print("Image tensor shape:", images.shape)
        print("Text preview:")
        print(clean_text(text)[:300], "...")
        print("Target class ID:", label)

    print("\n====================================\n")



# Main experiment

def main():
    print("\n===================================================")
    print(" MULTIMODAL STORY REASONING – EXPERIMENT PIPELINE ")
    print("===================================================\n")

    
    # Load config
    
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])
    print(f"[INFO] Device selected: {device}")
    print(f"[INFO] Hyperparameters: {cfg}\n")

    os.makedirs("results", exist_ok=True)

    
    # Load dataset
    
    print("[INFO] Loading dataset...")
    dataset = WrappedStoryDataset(
        split="train",
        max_frames=cfg["max_frames"]
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0
    )

    num_classes = len(dataset.label_map)

    print("[INFO] Dataset loaded successfully")
    print("[INFO] Total samples:", len(dataset))
    print("[INFO] Action label mapping:", dataset.label_map)
    print("[INFO] Number of target classes:", num_classes)

    
    # Dataset preview
    
    preview_dataset(dataset, num_samples=2)

    criterion = nn.CrossEntropyLoss()

    
    # BASELINE MODEL
    
    print("\n===================================================")
    print(" BASELINE MODEL: CONCATENATION FUSION ")
    print("===================================================")

    baseline = BaselineModel(cfg, num_classes).to(device)
    optimizer_b = torch.optim.Adam(
        baseline.parameters(),
        lr=cfg["learning_rate"]
    )

    start = time.time()
    train(
        baseline,
        loader,
        optimizer_b,
        criterion,
        device,
        log_file="results/baseline_loss.csv",
        epochs=cfg.get("epochs", 3),
        max_batches=30
    )
    baseline_time = time.time() - start

    print("\n[INFO] Evaluating Baseline Model...")
    b_acc = evaluate(baseline, loader, device, name="baseline")

    print(f"[RESULT] Baseline Accuracy: {b_acc:.4f}")
    print(f"[TIME] Baseline Training Time: {baseline_time:.2f}s")

    
    # PROPOSED MODEL
   
    print("\n===================================================")
    print(" PROPOSED MODEL: CROSS-MODAL ATTENTION ")
    print("===================================================")

    proposed = ProposedModel(cfg, num_classes).to(device)
    optimizer_p = torch.optim.Adam(
        proposed.parameters(),
        lr=cfg["learning_rate"]
    )

    start = time.time()
    train(
        proposed,
        loader,
        optimizer_p,
        criterion,
        device,
        log_file="results/proposed_loss.csv",
        epochs=cfg.get("epochs", 3),
        max_batches=30
    )
    proposed_time = time.time() - start

    print("\n[INFO] Evaluating Proposed Model...")
    p_acc = evaluate(proposed, loader, device, name="proposed")

    print(f"[RESULT] Proposed Accuracy: {p_acc:.4f}")
    print(f"[TIME] Proposed Training Time: {proposed_time:.2f}s")

    
    # PLOTS & COMPARISON
    
    print("\n===================================================")
    print(" RESULTS & ANALYSIS ")
    print("===================================================")

    print("[INFO] Plotting loss curves...")
    plot(
        "results/baseline_loss.csv",
        "results/baseline_loss.png",
        "Baseline Training Loss"
    )
    plot(
        "results/proposed_loss.csv",
        "results/proposed_loss.png",
        "Proposed Training Loss"
    )

    print("[INFO] Creating comparison table...")
    compare(
        b_acc,
        p_acc,
        "results/comparison.csv"
    )

    print("\n==================== FINAL SUMMARY ====================")
    print(f"Baseline Accuracy : {b_acc:.4f}")
    print(f"Proposed Accuracy : {p_acc:.4f}")

    if p_acc > b_acc:
        print(" Proposed model outperforms baseline")
    else:
        print(" No significant improvement observed")

    print("\nAll outputs saved inside the `results/` folder:")
    print(" - baseline_loss.csv / proposed_loss.csv")
    print(" - baseline_loss.png / proposed_loss.png")
    print(" - baseline_confusion.png / proposed_confusion.png")
    print(" - comparison.csv")

    print("\n EXPERIMENT COMPLETED SUCCESSFULLY\n")


if __name__ == "__main__":
    main()
