import time
import math
import random
import pickle

from dataset import ImageDataset
from model import SimpleCNN
from loss import cross_entropy_loss, accuracy
from optimizer import SGD


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
random.seed(42)


# -------------------------------------------------
# Gradient Norm Helper
# -------------------------------------------------
def grad_norm(tensor):
    total = 0.0
    for i in range(tensor.numel()):
        g = tensor.get_grad(i)
        total += g * g
    return math.sqrt(total)


# -------------------------------------------------
# Save / Load Weights
# -------------------------------------------------
def save_weights(model, path="best_model.pkl"):

    weights = {
        "conv_weight": [model.conv_weight.get_item(i)
                        for i in range(model.conv_weight.numel())],

        "conv_bias": [model.conv_bias.get_item(i)
                      for i in range(model.conv_bias.numel())],

        "fc_weight": [model.fc_weight.get_item(i)
                      for i in range(model.fc_weight.numel())],

        "fc_bias": [model.fc_bias.get_item(i)
                    for i in range(model.fc_bias.numel())],
    }

    with open(path, "wb") as f:
        pickle.dump(weights, f)


# -------------------------------------------------
# Train Function
# -------------------------------------------------
def train(
    dataset_path,
    epochs=10,
    batch_size=64,
    lr=0.005,
    weight_decay=1e-4,
    val_split=0.2,
    patience=5
):

    print("Loading dataset...")
    dataset = ImageDataset(dataset_path)

    print("Dataset size:", len(dataset))
    print("Dataset loading time (s):",
          dataset.get_load_time())

    # -----------------------
    # Train/Validation Split
    # -----------------------
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split_idx = int(len(dataset) * (1 - val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    print("Train size:", len(train_indices))
    print("Validation size:", len(val_indices))

    # -----------------------
    # Build Model
    # -----------------------
    print("Building model...")
    model = SimpleCNN(num_classes=dataset.num_classes)


    print("Total parameters:",
          model.count_parameters())

    print("Total MACs per forward pass:",
          model.compute_macs())

    print("Total FLOPs per forward pass:",
          model.compute_flops())

    optimizer = SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_acc = 0.0
    patience_counter = 0

    print("Starting training...\n")

    for epoch in range(epochs):

        # =====================
        # TRAINING LOOP
        # =====================
        epoch_loss = 0.0
        epoch_acc = 0
        total_samples = 0

        start_time = time.time()

        for start_idx in range(0,
                               len(train_indices),
                               batch_size):

            batch_ids = train_indices[
                start_idx:start_idx+batch_size
            ]

            images, labels = dataset.get_batch_by_indices(batch_ids)

            for x, label in zip(images, labels):

                optimizer.zero_grad()

                logits = model.forward(x)

                loss = cross_entropy_loss(
                    logits,
                    label
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.get_item(0)
                epoch_acc += accuracy(
                    logits,
                    label
                )

                total_samples += 1

        train_loss = epoch_loss / total_samples
        train_acc = epoch_acc / total_samples

        # =====================
        # VALIDATION LOOP
        # =====================
        val_loss = 0.0
        val_acc = 0
        val_samples = 0

        for start_idx in range(0,
                               len(val_indices),
                               batch_size):

            batch_ids = val_indices[
                start_idx:start_idx+batch_size
            ]

            images, labels = dataset.get_batch_by_indices(batch_ids)

            for x, label in zip(images, labels):

                logits = model.forward(x)

                loss = cross_entropy_loss(
                    logits,
                    label
                )

                val_loss += loss.get_item(0)
                val_acc += accuracy(
                    logits,
                    label
                )

                val_samples += 1

        val_loss /= val_samples
        val_acc /= val_samples

        end_time = time.time()

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("  Train Loss:", train_loss)
        print("  Train Accuracy:", train_acc)
        print("  Val Loss:", val_loss)
        print("  Val Accuracy:", val_acc)
        print("  Epoch time (s):",
              end_time - start_time)
        print("----------------------------------")

        # =====================
        # Early Stopping
        # =====================
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_counter = 0

            save_weights(model)
            print("Saved best model.\n")

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training finished.")
    print("Best Validation Accuracy:",
          best_val_acc)

if __name__ == "__main__":
    train("C:/mine/ASSIGNMENTS/DL/assign1/data_1",
          epochs=6,
          batch_size=64,
          lr=0.005,
          weight_decay=1e-4)