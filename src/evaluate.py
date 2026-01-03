import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(model, loader, device, name):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(device)
            logits = model(images, texts)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred += preds
            y_true += labels.tolist()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"results/{name}_confusion.png")
    plt.close()

    return acc


def compare(b_acc, p_acc, out_file):
    df = pd.DataFrame({
        "Model": ["Baseline", "Proposed"],
        "Validation Accuracy": [b_acc, p_acc]
    })
    df.to_csv(out_file, index=False)
