from pathlib import Path

import torch
import json
import matplotlib.pyplot as plt

from train import SmallCNN
from train import split_data


# Find the saved weights relative to this script. 

project_folder = Path(__file__).resolve().parent
demo_output = project_folder / "results" / "demo" # save preditions and plots to this folder
demo_output.mkdir(parents=True, exist_ok=True)
model_path = project_folder / "results" / "local" / "cnn_fp32.pt"

# Build the same architecture used during training.

model = SmallCNN() # creates the layers with their initial parameters (weights and biases)

# Read the learned weights and biases.

saved_weights = torch.load( # reads the saved weights and biases onto the CPU
    model_path,
    map_location="cpu",
    weights_only=True,
)

# Put the learned values into the model.

model.load_state_dict(saved_weights) # replaces the intiial values with the learned values

# Set the model to evaluation mode.

model.eval() # sets every layer to evaluation mode (e.g., dropout and batch normalization layers 
## behave differently during training and evaluation)

print("Trained CNN loaded successfully.")

# Read the report from the same training run.
report_path = project_folder / "results" / "local" / "report.json"
report = json.loads(report_path.read_text())

# Prepare images using the original preprocessing.
x, y, _, _, _ = split_data(report["seed"])

# Select the first image from the saved test split.
image_index = report["split_indices"]["test"][0]
sample = x[image_index:image_index + 1]

# Make a prediction without tracking gradients.
with torch.inference_mode():
    logits = model(sample)
    predicted_digit = logits.argmax(dim=1).item()

actual_digit = y[image_index].item()

print("Input shape:", tuple(sample.shape))
print("Output shape:", tuple(logits.shape))
print("Predicted digit:", predicted_digit)
print("Actual digit:", actual_digit)

# Select the image's grayscale channel for display.
image = sample[0, 0].numpy()

plt.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
plt.title(f"Predicted: {predicted_digit} | Actual: {actual_digit}")
plt.axis("off")
plt.savefig(
    demo_output / "single_prediction.png",
    dpi=150,
    bbox_inches="tight",
) 

plt.show()

# Run predictions on the complete saved test split.
test_indices = report["split_indices"]["test"]
test_images = x[test_indices]
test_labels = y[test_indices]

with torch.inference_mode():
    test_predictions = model(test_images).argmax(dim=1) # selects position of the largest logit for each image

# Find positions where the prediction differs from the label.
wrong_positions = (
    test_predictions != test_labels
).nonzero(as_tuple=True)[0]

print("Test images:", len(test_indices))
print("Incorrect predictions:", len(wrong_positions))

# Display up to six mistakes.
if len(wrong_positions) > 0:
    count = min(6, len(wrong_positions))
    fig, axes = plt.subplots(1, count, figsize=(3 * count, 3), squeeze=False)

    for ax, position in zip(axes[0], wrong_positions[:count].tolist()):
        image = test_images[position, 0].numpy()
        predicted = test_predictions[position].item()
        actual = test_labels[position].item()

        ax.imshow(
            image, cmap="gray", vmin=0, vmax=1,
            interpolation="nearest"
        )
        ax.set_title(f"Predicted: {predicted}\nActual: {actual}")
        ax.axis("off")

    fig.suptitle("CNN prediction mistakes")
    plt.tight_layout()
    fig.savefig(
    demo_output / "prediction_mistakes.png",
    dpi=150,
    bbox_inches="tight",
)
    plt.show()
else:
    print("The model correctly classified every test image.")
    # Find test images labeled 6 that the model predicted as 8.
matches = (
    (test_labels == 6) & (test_predictions == 8)
).nonzero(as_tuple=True)[0]

if len(matches) > 0:
    # Select the first matching mistake.
    position = matches[0].item()
    sample = test_images[position:position + 1]

    with torch.inference_mode():
        logits = model(sample)
        probabilities = torch.softmax(logits, dim=1)

    print("\nActual 6, predicted 8: model scores")

    for digit in range(10):
        score = logits[0, digit].item()
        probability = probabilities[0, digit].item()

        print(
            f"Digit {digit}: "
            f"logit = {score:.3f}, "
            f"softmax = {probability:.2%}"
        )
else:
    print("No actual-6 / predicted-8 example found.")
