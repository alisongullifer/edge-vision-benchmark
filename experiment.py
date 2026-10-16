import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from train import SmallCNN, predict, split_data


OUTPUT = Path("results/day6")
SEED = 42
EPOCHS = 30


class WiderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 32 feature maps × 2 rows × 2 columns = 128.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def train_variant(name, model_class, x, y, train, val):
    # Reset randomness before training each architecture.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # A separate generator gives both models the same batch order.
    batch_generator = torch.Generator().manual_seed(SEED)

    best_accuracy = -1.0
    best_epoch = None
    best_state = None
    history = []

    train_indices = torch.tensor(train)
    target = y[val].numpy()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        order = train_indices[
            torch.randperm(
                len(train_indices),
                generator=batch_generator,
            )
        ]

        total_loss = 0.0

        for batch in order.split(64):
            optimizer.zero_grad()

            logits = model(x[batch])
            loss = nn.functional.cross_entropy(logits, y[batch])

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        model.eval()
        predictions = predict(model, x[val])
        accuracy = float(np.mean(predictions == target))

        history.append({
            "epoch": epoch,
            "training_loss": total_loss / len(train),
            "validation_accuracy": accuracy,
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"{name} | epoch {epoch:02d}/{EPOCHS} | "
            f"validation accuracy: {accuracy:.2%}"
        )

    # Restore the checkpoint chosen using validation accuracy.
    model.load_state_dict(best_state)
    model.eval()

    predictions = predict(model, x[val])
    mistakes = []

    for position, dataset_index in enumerate(val):
        actual = int(target[position])
        predicted = int(predictions[position])

        if actual != predicted:
            mistakes.append({
                "dataset_index": int(dataset_index),
                "actual_label": actual,
                "predicted_label": predicted,
            })

    matrix = confusion_matrix(
        target,
        predictions,
        labels=list(range(10)),
    )

    confusion_pairs = []

    for actual in range(10):
        for predicted in range(10):
            count = int(matrix[actual, predicted])

            if actual != predicted and count:
                confusion_pairs.append({
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "count": count,
                })

    confusion_pairs.sort(key=lambda row: -row["count"])

    with (OUTPUT / f"{name}_validation_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_index",
                "actual_label",
                "predicted_label",
            ],
        )
        writer.writeheader()
        writer.writerows(mistakes)

    torch.save(
        best_state,
        OUTPUT / f"{name}_best_validation.pt",
    )

    return {
        "model": name,
        "parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "selected_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "validation_correct": int(
            np.sum(predictions == target)
        ),
        "validation_count": len(val),
        "validation_error_count": len(mistakes),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": list(range(10)),
        "confusion_matrix_axes": {
            "rows": "actual labels",
            "columns": "predicted labels",
        },
        "confusion_pairs": confusion_pairs,
        "training_history": history,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    x, y, train, val, _ = split_data(SEED)

    # Confirm these are the original training and validation splits.
    original = json.loads(
        Path("results/local/report.json").read_text()
    )

    if original["seed"] != SEED:
        raise ValueError("Seed does not match the original report.")

    if (
        train.tolist() != original["split_indices"]["train"]
        or val.tolist() != original["split_indices"]["validation"]
    ):
        raise ValueError("Data splits do not match the original report.")

    results = []

    for name, model_class in [
        ("original_cnn", SmallCNN),
        ("wider_cnn", WiderCNN),
    ]:
        results.append(
            train_variant(
                name, model_class, x, y, train, val
            )
        )

    report = {
        "seed": SEED,
        "epochs_per_model": EPOCHS,
        "threads": 1,
        "batch_size": 64,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "evaluation_split": "validation",
        "test_evaluated": False,
        "checkpoint_selection": (
            "Highest validation accuracy; earliest epoch wins ties"
        ),
        "results": results,
    }

    path = OUTPUT / "validation_comparison.json"
    path.write_text(json.dumps(report, indent=2))

    print("\nVALIDATION COMPARISON")

    for result in results:
        print(
            f"{result['model']}: "
            f"{result['parameter_count']:,} parameters | "
            f"validation accuracy "
            f"{result['validation_accuracy']:.2%} | "
            f"{result['validation_error_count']} errors | "
            f"selected epoch {result['selected_epoch']}"
        )

        print("Most frequent validation confusions:")

        for pair in result["confusion_pairs"][:5]:
            print(
                f"  Actual {pair['actual_label']} -> "
                f"predicted {pair['predicted_label']}: "
                f"{pair['count']} image(s)"
            )

    print(f"\nSaved results to {path}")
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()