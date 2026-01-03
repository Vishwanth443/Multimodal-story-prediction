import csv
import torch

def train(model, loader, optimizer, criterion, device, log_file, epochs=3, max_batches=30):
    model.train()

    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "batch", "loss"])

    for epoch in range(epochs):
        print(f"\n[INFO] Epoch {epoch+1}")
        total_loss = 0

        for i, (images, texts, labels) in enumerate(loader):
            if i >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images, texts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([epoch+1, i+1, loss.item()])

            print(f"[TRAIN] Batch {i+1} | Loss {loss.item():.4f}")

        print(f"[EPOCH DONE] Avg Loss: {total_loss/max_batches:.4f}")
