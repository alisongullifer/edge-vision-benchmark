# Project Learning Notes

## Environment Setup and Initial Experiment
-Day 1 (October 7th, 2026) 

### Goal

To set up the development environment, run the initial experiment,
review each function, record software versions, and explain the
inputs, labels, and training, validation, and test sets.

### Work Completed

- Created a project-specific Python virtual environment.
- Installed PyTorch, NumPy, and scikit-learn.
- Passed both pipeline tests.
- Updated quantization-engine selection to support the available
  oneDNN backend.
- Completed a 30-epoch training and evaluation run.
- Reviewed the functions and generated an experiment report for Day 1.

### Software Versions and Settings

- Operating system: Windows 11
- Python: 3.13.14
- PyTorch: 2.14.1+cpu
- NumPy: 2.5.3
- scikit-learn: 1.9.1
- Quantization backend: onednn
- Random seed: 42
- CPU threads: 1

### Inputs and Labels

The project uses scikit-learn's digits dataset, which contains
1,797 grayscale images of handwritten digits from 0 through 9.

Each input is an 8 x 8 image represented by 64 pixel brightness
values. Its label is the correct digit associated with that image.

The program divides pixel values by 16 to normalize their range
from 0-16 to 0-1. This changes the numerical scale while preserving
the image's brightness pattern.

A channel dimension represents the single grayscale channel.
The complete input shape is (1797, 1, 8, 8), representing the number
of images, channels, rows, and columns.

### Training, Validation, and Test Sets

- Training: 1,077 images used to calculate loss and update weights.
- Validation: 360 images used to check progress and select a checkpoint.
- Test: 360 images used to evaluate the selected model.

The split preserves approximately similar digit proportions in each
group. Validation results guide model selection, while test results
provide a separate final evaluation.

### Function Review

- SmallCNN.__init__: Defines the model's layers.
- forward: Processes images and returns ten digit scores per image.
- split_data: Loads, normalizes, reshapes, and divides the dataset.
- predict: Selects the highest-scoring digit for each image.
- latency: Measures CPU model execution time.
- state_bytes: Measures serialized model-state size.
- run: Coordinates training, checkpoint selection, quantization,
  evaluation, benchmarking, and saving results.

The tests check data-split integrity, input shape and range, output
shape, and finite gradients. The experiment also checks that reloading
the FP32 model preserves its predictions.

### Initial Results

Training ran for 30 epochs. Epoch 18 was selected with 97.22%
validation accuracy. Epoch 30 tied that score, but the script kept
the first checkpoint that achieved the best accuracy.

Logistic-regression baseline:
- Test accuracy: 95.56%

Original FP32 CNN:
- Test accuracy: 95.83%
- Serialized model-state size: 27,597 bytes
- Median CPU inference time: 0.0807 milliseconds

CNN with INT8 fully connected layers:
- Test accuracy: 96.11%
- Serialized model-state size: 14,783 bytes
- Median CPU inference time: 0.2684 milliseconds

### Findings

Quantization reduced saved model-state size by approximately 46.4%.
However, median inference time increased to approximately 3.33 times
the original model's time on this CPU.

The quantized model correctly classified one additional test image.
This small difference does not establish a general accuracy improvement.

The main finding is a measured tradeoff: reduced storage did not
produce faster inference in this experiment.

### Limitations

Only the fully connected Linear layers were quantized. Convolution
layers remained FP32.

Latency measurements excluded preprocessing and model loading.
Serialized state size does not represent total runtime memory use.

The project uses a small educational dataset.

### Contributions and Next Step

Using AI-provided starter code and guidance, I configured the local
environment, applied the backend-selection correction, passed the
checks, and reproduced the experiment on my laptop.

Day 2 will focus on weights, biases, ReLU, cross-entropy loss,
gradients, backpropagation, and parameter updates.

--------------------------------------------------------------------

## Model Structure and How Training Works
- Day 2 (October 8th, 2026)

### Goal

To understand neurons, weights, biases, ReLU, cross-entropy,
gradient descent, and backpropagation using the project's CNN.
Trace one batch through the model and explain why training
updates weights and biases.

### Work Completed

- Reviewed the SmallCNN architecture.
- Clarified the difference between pixels, images, and batches.
- Reviewed how filters produce feature maps.
- Explained how ReLU, pooling, and flattening process those maps.
- Traced a batch through the model to its ten output scores.
- Reviewed loss calculation, gradients, backpropagation, and Adam.
- Completed guided questions and calculation examples.

### Inputs and Batches

Each image contains 8 x 8 = 64 pixel brightness values.
A batch of 64 contains 64 separate images, each with its own
pixel values and correct label.

Pixel values come from scikit-learn's digits dataset. They are
normalized from 0-16 to 0-1 before entering the model.

A batch is a group of images processed together.

The batch input shape is (64, 1, 8, 8), representing images,
channels, rows, and columns.

### Neurons, Weights, and Biases

A nueron is a mathematical unit defined by its connections, weights, 
and bias that produces one output value.

A weight is a learned multiplier attached to an input connection.

A bias is a learned number added to the weighted sum.

A neuron calculates a weighted sum of its inputs and adds a bias.
An activation function can then transform the result.

Weights determine how each input contributes to the calculation.
A bias shifts the result independently of the inputs.

PyTorch initializes the model's parameters when its layers are
created. Training adjusts weights and biases to try to reduce loss.

### Filters and Feature Maps

A filter, also called a kernel, is a small set of learned weights.
A feature map is the grid results that filter produces.

The first convolution uses eight filters, each examining a
3 x 3 region, or neighborhood, of the grayscale image at a time.

Each filter moves across overlapping regions using the same
weights. Its calculated responses form a feature map at each position.

One image produces eight feature maps in the first convolution.
The second convolution combines information from those eight
maps to produce sixteen new maps per image.

Original pixels represent brightness, feature-map values are calculated 
filter responses.


### ReLU, Pooling, and Flattening

ReLU changes negative values to zero and keeps positive values.
It introduces nonlinearity and has no learned parameters.

Max pooling keeps the largest value from each 2 x 2 region.
It reduces each map's rows and columns without changing the
number of maps.

Flatten rearranges each image's feature maps into a list.
It does not change their values.

### Batch Trace

For a batch of 64 images:

- Input: (64, 1, 8, 8).
- First convolution and ReLU: (64, 8, 8, 8).
- First pooling: (64, 8, 4, 4).
- Second convolution and ReLU: (64, 16, 4, 4).
- Second pooling: (64, 16, 2, 2).
- Flatten: (64, 64).
- First Linear layer and ReLU: (64, 64).
- Final Linear layer: (64, 10).

The sixteen 2 x 2 maps contain 64 feature values per image.
These are calculated features, distinct from the original pixels.

The final Linear layer produces ten logits per image, one for
each digit from 0 through 9. Each output neuron receives all
64 values from the preceding layer.

### Cross-Entropy Loss

Logits are raw scores, not probabilities. The highest score
determines the predicted digit.

Cross-entropy compares the logits with the correct labels.
Assigning more probability to the correct digit produces a
lower loss.

A correct prediction can still have positive loss. Accuracy
checks whether the correct digit has the highest score, while
loss also reflects the probability assigned to that digit.

The training code averages the individual losses across the batch.
Labels are used in this calculation, not as inputs to the CNN.

### Gradients and Parameter Updates

A gradient describes how a small change in a parameter would
affect the loss locally.

Backpropagation calculates gradients by working backward
through the model's connected calculations.

Basic gradient descent subtracts the learning rate multiplied
by the gradient from the current parameter value.

The project uses Adam, an optimizer that uses gradients and
their history to calculate parameter adjustments.

The training steps are:

- optimizer.zero_grad(): Clears previous gradients.
- model(x[batch]): Calculates the batch's output scores.
- cross_entropy(...): Calculates loss using the correct labels.
- loss.backward(): Calculates gradients.
- optimizer.step(): Updates weights and biases using Adam.

### Findings

Training changes weights and biases because they control how
the model converts image inputs into predictions.

The image pixels remain unchanged. Updated parameters can
produce different feature maps and output scores when the same
image passes through the model again.

Convolution and Linear layers have learned parameters.
ReLU, pooling, and flattening follow fixed rules.

### Limitations

Day 2 focused on understanding the supplied model and tracing
its calculations. No new training run, code changes, or benchmark
measurements were completed.

A parameter update aims to reduce loss but does not guarantee
better predictions for every image or improved test accuracy.

### Contributions and Next Step

Using guided explanations, I reviewed the model's architecture,
worked through examples, and traced how a batch leads to loss,
gradients, and parameter updates.

The next step is to practice explaining this training process
independently and continue with the next program-plan task.

----------------------------------------------------------------

## Convolution and Tensor Shapes
- Day 3 (October 9th, 2026)

### Goal

To understand convolution, shared kernels, pooling, channels,
logits, and tensor shapes. Document the shape after each layer
and explain why a CNN is suitable for images.

### Work Completed

- Reviewed how convolution filters process local neighborhoods.
- Explained weight sharing across image locations.
- Reviewed how filters combine information across input channels.
- Calculated convolution output size with and without padding.
- Documented the tensor shape after every layer.
- Reviewed pooling, flattening, and the final digit scores.
- Explained why CNNs are useful for recognizing image patterns.

### Convolution and Shared Kernels

Convolution is the operation that applies a filter's weight to 
neighborhoods of an image to produce a feature map.

A convolution filter contains learned weights that process
small neighborhoods of nearby values.

Weight sharing means that each convolution filter reuses the same
kernel weights at every spatial position where it is applied.

The first convolution receives one grayscale channel and uses
eight filters to produce eight feature maps per image.

The second convolution receives eight feature maps and uses
sixteen filters to produce sixteen new maps. Each output filter
combines information from all eight incoming channels.

A filter in the first layer has 3 × 3 = 9 weights.
A filter in the second layer has 8 × 3 × 3 = 72 weights.
Each output filter also has one bias.

### Padding and Stride

Padding adds a border of zeros around the input during convolution.
Stride determines how far the filter moves between calculations.

The model uses 3 × 3 kernels, padding of one, and stride of one.
These settings preserve the input's rows and columns.

For an input width of eight:

Output width = (8 + 2 × 1 - 3) / 1 + 1 = 8.

Without padding, the same convolution would produce a width
and height of six.

### Channels and Feature Maps

The original image has one channel containing brightness values.

Convolution output channels contain calculated feature maps.
These values represent filter responses rather than original
pixel brightness.

One output filter produces one feature map per image. Multiple
filters produce multiple representations of the same image,
while the number of images stays unchanged.

### Pooling and Flattening

Max pooling keeps the largest value from each 2 × 2 region.
In this model, it halves the rows and columns while preserving
the image count and channel count.

Flatten rearranges the feature maps into one list per image
without changing their values.

Sixteen feature maps of size 2 × 2 produce:
16 × 2 × 2 = 64 feature values per image.

The next Linear layer's input size must match this value count.

### Shape After Each Layer

For a full batch of 64 images, the shape order before Flatten
is images, channels, rows, and columns.

- Input: (64, 1, 8, 8).
- First convolution: (64, 8, 8, 8).
- First ReLU: (64, 8, 8, 8).
- First pooling: (64, 8, 4, 4).
- Second convolution: (64, 16, 4, 4).
- Second ReLU: (64, 16, 4, 4).
- Second pooling: (64, 16, 2, 2).
- Flatten: (64, 64).
- First Linear layer: (64, 64).
- Third ReLU: (64, 64).
- Final Linear layer: (64, 10).

ReLU changes negative values to zero but preserves shape.

The first dimension represents the batch size throughout.
One image produces final shape (1, 10), while five images
produce final shape (5, 10).

### Logits

The final Linear layer produces ten raw scores per image,
one for each digit from 0 through 9.

These scores are called logits and are not probabilities.
The highest score determines the predicted digit.

### Why a CNN Is Suitable for Images

Neighboring pixels form meaningful patterns such as strokes,
edges, and curves. Convolution examines these local patterns.

Reusing filters across locations allows the CNN to detect
useful patterns throughout an image with fewer parameters
than separate weights for every location.

Later layers combine earlier features to help identify the digit.
An individual filter does not identify the whole digit by itself.

### Findings

Different operations affect different parts of the tensor:

- Convolution changes the channel count in this model.
- Padding and stride help determine spatial output size.
- ReLU changes values without changing shape.
- Pooling reduces rows and columns.
- Flatten rearranges features into a list per image.
- Linear layers calculate a specified number of output values.

The image count remains unchanged throughout the forward pass.

A tensor is a numerical data structure organized along one or 
more dimensions. In PyTorch, tensors store inputs, weights,
intermediate results, and outputs. 

### Limitations

Day 3 focused on architecture understanding and shape calculations.
No new code changes, training runs, or benchmarks were completed.

Pooling discards spatial detail. Weight sharing helps detect
patterns at different locations but does not guarantee identical
predictions when an image shifts.

### Contributions and Next Step

Using guided explanations and calculation exercises, I documented
every layer's output shape and explained how the CNN processes
image patterns.

The next step is to apply this understanding during the Day 4
visual prediction demo described in the program plan.

------------------------------------------------------------------

## Visual Prediction Demo and Error Analysis
- Day 4 (October 10, 2026)

### What I completed

- Created demo.py to load my trained CNN.
- Used the saved test split to make predictions.
- Displayed a digit image with its predicted and actual labels.
- Displayed up to six incorrect predictions.
- Examined logits and softmax values for one mistake.
- Saved prediction images for my project README.

### Restoring the trained model

SmallCNN() creates the model’s architecture with initial
parameter values.

cnn_fp32.pt contains the learned weights and biases saved
during training.

model.load_state_dict(saved_weights) loads those learned values
into the architecture.

Both the architecture and the learned values are needed to
restore the trained CNN.

Importing SmallCNN from train.py does not start training because
the training command is inside the if __name__ == "__main__":
block.

### Inference

Inference means using a trained model to make a prediction.

During training, the model updates its weights and biases to
learn from examples. During inference, it uses the learned
values without updating them.

In my demo, model(sample) passes an image through the CNN and
produces ten raw digit scores.

Comparing a prediction with the actual label does not teach
the model or update its weights.

### Evaluation mode and gradient tracking

model.eval() sets the model to evaluation mode. This controls
the behavior of layers such as Dropout and BatchNorm if they
are present. My SmallCNN does not contain those layers.

torch.inference_mode() disables gradient tracking and related
overhead while making predictions.

Evaluation mode and inference mode serve different purposes.

### Input and output shapes

The input for one image has shape (1, 1, 8, 8):

- 1 image in the batch.
- 1 grayscale channel.
- 8 rows.
- 8 columns.

The output has shape (1, 10):

- 1 image in the batch.
- 10 logits, one for each digit from 0 through 9.

Keeping the batch dimension allows the image to pass through
the model correctly.

### Selecting the predicted digit

A logit is a raw model output score before conversion into
a probability.

logits.argmax(dim=1) selects the position of the highest score
across the ten digit classes.

Because the classes are digits 0 through 9, that position
gives the predicted digit.

The model receives the image as input. The actual label is
used afterward to check the prediction.

### Displaying an image

sample[0, 0] selects the first image and its grayscale channel.

This leaves an (8, 8) grid of pixel values that Matplotlib
can display.

The normalized pixel values range from 0 to 1. In the
grayscale display, 0 appears black and 1 appears white.

plt.show() opens the image window and, in my run, waits until
I close it before continuing.

### My test results

The CNN made predictions on 360 test images.

Correct predictions: 345
Incorrect predictions: 15

Accuracy = 345 / 360 × 100 = 95.83%

Error rate = 15 / 360 × 100 = 4.17%

These results describe performance on this project’s held-out
test split. They do not establish performance on other
image sources.

### Error-analysis observations

Two displayed images had actual labels of 9, but the model
predicted 4. Both images also looked like 4s to me.

These count as errors against the dataset labels. My visual
interpretation does not prove that the labels are incorrect.

Another image had an actual label of 6, but the model
predicted 8. That image looked like a 6 to me.

This shows that the CNN can misclassify an image that a
person finds recognizable.

Inspecting mistakes helps me identify possible patterns, but
the predicted label alone does not explain exactly why the
model made an error.

### Softmax

Softmax converts logits into values between 0 and 1 that
sum to 1 across the digit classes.

It preserves their ranking, so the digit with the highest
logit also has the highest softmax value.

Softmax describes the model’s relative preference among the
classes. A high value does not guarantee that a prediction
is correct.

### Actual 6, predicted 8

For the inspected mistake:

- Digit 8 had a logit of 2.791 and a softmax value of 89.69%.
- Digit 6 had a logit of -0.229 and a softmax value of 4.38%.

The CNN strongly favored 8 even though the actual label was 6.

An 80% acceptance threshold would not have prevented this
mistake because 89.69% exceeds that threshold.

A displayed softmax value of 0.00% can result from rounding
a very small positive value.

### Saved outputs

results/demo/single_prediction.png shows the correctly
predicted 5.

results/demo/prediction_mistakes.png shows up to six incorrect
predictions with their actual labels.

These images provide visual evidence for the demo section
of my README.

### What I can explain

I can explain how to restore a trained CNN, prepare an image
for inference, select a predicted digit, compare predictions
with labels, and inspect classification mistakes.

I can also explain why inference does not update weights
and why a high softmax value can accompany an incorrect
prediction.

--------------------------------------------------------------

## Quantization and Benchmarking
- Day 5 (Octover 12, 2026)

### Goal

To understand quantization, repeat inference timing without
retraining, and record background activity during benchmarking.

### Work Completed

- Created benchmark.py to load the saved model checkpoints.
- Repeated inference timing three times without retraining.
- Recorded open applications, power settings, and CPU usage.
- Reversed model timing order during the second run.
- Compared FP32 and INT8 Linear inference times.
- Explained reduced saved size versus slower inference.

### What Quantization Means

Quantization represents numerical values using fewer bits.

FP32 uses 32 bits, or four bytes, per weight.
INT8 uses eight bits, or one byte, per quantized weight.

Quantized weights also require scaling information. Quantization
can reduce saved model-state size, but it does not guarantee
faster inference.

### Which Layers Were Quantized

Only the two fully connected Linear layers were quantized.

The two convolution layers remained FP32.

The code selects Linear layers using {nn.Linear} and targets
quantized weights using dtype=torch.qint8.

The correct label is “CNN with INT8 Linear layers.”
The model is not a fully INT8 CNN.

### Why a Smaller Model Can Run Slower

Saved model-state size measures storage.
Inference time measures how long model execution takes.

Dynamic quantization converts incoming activation values during
inference, which adds extra work.

For a small model processing one image, this overhead may
outweigh the benefit of INT8 calculations.

This is a possible explanation for my results. Timing alone
does not establish the exact cause.

### Repeating Timing Without Retraining

The benchmark loaded cnn_fp32.pt and cnn_linear_int8.pt.

Loading the saved checkpoints kept each variant’s learned
values fixed across the benchmark runs.

Retraining could change the weights and introduce another
difference between runs.

No training or weight updates occurred during benchmarking.

### Benchmark Settings

- Device: CPU.
- CPU threads: 1.
- Quantization backend: onednn.
- Batch size: 1 image.
- Test image index: 618.
- Warmup passes: 30 per variant.
- Timing rounds: 5 per variant.
- Measurements per round: 200.

Each variant had 1,000 timed forward passes per run.

Warmup passes occurred before timing. Model loading and
image preparation were outside the timed section.

### Background Activity

During all three runs, VS Code and Edge were open.
No downloads were noticed.

The laptop was plugged in and used Balanced power mode.

CPU usage observed before each run:

- Run 1: approximately 5%.
- Run 2: approximately 8%.
- Run 3: approximately 6%.

Background applications can compete for CPU resources and
affect timing.

These observations describe conditions before each run.
They do not prove background activity caused timing differences.

### Benchmark Results

Run 1 — FP32 timed first:

- FP32 median: 0.0753 milliseconds.
- INT8 Linear median: 0.1975 milliseconds.
- INT8 Linear inference took approximately 2.62 times as long.

Run 2 — INT8 Linear timed first:

- FP32 median: 0.0775 milliseconds.
- INT8 Linear median: 0.2062 milliseconds.
- INT8 Linear inference took approximately 2.66 times as long.

Run 3 — FP32 timed first:

- FP32 median: 0.0766 milliseconds.
- INT8 Linear median: 0.2038 milliseconds.
- INT8 Linear inference took approximately 2.66 times as long.

The CNN with INT8 Linear layers remained slower in all
three runs, including when it was timed first.

### Median and 95th Percentile

The median is the middle measured inference time.
Half the measurements fall at or below it.

The 95th percentile is the time at or below which
approximately 95% of the measurements fall.

These measurements describe typical timing and slower execution
within the benchmark.

### Saved Size Versus Runtime Memory

The original experiment measured serialized model-state size:

- FP32 CNN: 27,597 bytes.
- CNN with INT8 Linear layers: 14,783 bytes.
- Saved size reduction: approximately 46.4%.

These sizes were measured during the original experiment.
Day 5 repeated timing rather than measuring size again.

Saved size does not represent total runtime memory use.

Runtime memory also includes inputs, intermediate results,
temporary buffers, and other execution resources.

Total runtime memory was not measured.

### Findings

The trained FP32 CNN had faster inference across all three
benchmark repeats.

The CNN with INT8 Linear layers took approximately 2.6–2.7
times as long, despite its smaller saved state.

Reducing storage and improving inference speed are separate
outcomes that must be measured.

### Limitations

Only Linear layers were quantized; convolutions remained FP32.

Measurements used one image on my laptop’s CPU.

Timing excluded preprocessing and model loading.

Background activity was recorded manually rather than monitored
throughout each benchmark.

The results do not establish performance on Qualcomm hardware
or identify the exact cause of slower quantized inference.

### What I Can Explain

I can explain how quantization reduces numerical precision
and can reduce saved model-state size.

I can identify which layers were quantized and explain
why the whole CNN should not be labeled INT8.

I can explain why saved checkpoints support consistent
timing comparisons without retraining.

I can interpret median inference time, the 95th percentile,
and recorded background activity.

I can explain why smaller saved size does not guarantee
faster inference or lower total runtime memory use.

----------------------------------------------------------  

## Error Analysis and Architecture Comparison
- Day 6 (October 13, 2026)

### Goal

To analyze validation mistakes and compare the original CNN
with a wider architecture using validation accuracy and
parameter count, without tuning on test results.

### Work Completed

- Created experiment.py to train and compare two architectures.
- Increased convolution channel counts in the wider model.
- Trained both models for 30 epochs using consistent settings.
- Selected each checkpoint using validation accuracy.
- Compared validation accuracy and parameter count.
- Examined individual validation mistakes.
- Saved results in results/architecture_comparison.
- Did not evaluate the test set during this experiment.

### Architecture Changes

The original CNN uses eight output channels in the first
convolution and sixteen in the second.

The wider CNN uses sixteen output channels in the first
convolution and thirty-two in the second.

More channels produce more feature maps, giving the model
more capacity to learn different image patterns.

Increasing channels adds learned weights and biases.
It does not guarantee better predictions.

### Flatten and Linear Input Size

After the second pooling operation, each feature map
contains two rows and two columns.

Original CNN:

16 × 2 × 2 = 64 values per image.

Wider CNN:

32 × 2 × 2 = 128 values per image.

Flatten rearranges these values into one list per image.

The original model uses nn.Linear(64, 64).
The wider model uses nn.Linear(128, 64).

The final Linear layer still produces ten scores because
both models predict digits zero through nine.

### Parameters

A parameter is a learned value, such as a weight or bias.

- Original CNN: 6,058 parameters.
- Wider CNN: 13,706 parameters.
- Additional parameters: 7,648.
- Parameter increase: approximately 126%.
- The wider CNN has approximately 2.26 times the parameters.

ReLU, max pooling, and Flatten do not add learned parameters.

### Training Settings

- Random seed: 42.
- Training images: 1,077.
- Validation images: 360.
- Epochs: 30 per model.
- Batch size: 64.
- Optimizer: Adam.
- Learning rate: 0.003.
- CPU threads: 1.

Both models used the same training and validation splits,
epoch budget, and batch order.

Each architecture was trained from a fresh initialization.
Their weights were not identical because their shapes differ.

The experiment used a separate generator for batch order.
This differs from the original training script, so the new
original-CNN checkpoint need not match the earlier checkpoint.

### Why Use Validation Results

Validation results guide development decisions, such as
choosing channel counts and selecting checkpoints.

Using test scores to make those decisions would allow
test results to influence the model design.

The test set should remain separate from architecture tuning.

This experiment evaluated validation performance only.
Earlier project work had already evaluated the original
model on the test set.

### Checkpoint Selection

Each model kept the checkpoint with its highest validation
accuracy during the thirty training epochs.

If multiple epochs tied, the first matching epoch was kept.

Both models selected epoch 29.

The comparison used these selected checkpoints rather than
automatically using the final epoch.

### Validation Results

Original CNN:

- Validation accuracy: 97.22%.
- Correct predictions: 350 out of 360.
- Incorrect predictions: 10.
- Parameters: 6,058.
- Selected epoch: 29.

Wider CNN:

- Validation accuracy: 98.06%.
- Correct predictions: 353 out of 360.
- Incorrect predictions: 7.
- Parameters: 13,706.
- Selected epoch: 29.

The wider model correctly classified three additional images.

Using the unrounded scores, validation accuracy improved
by approximately 0.83 percentage points.

### What Error Analysis Means

Error analysis examines incorrect predictions to identify
patterns and compare model behavior.

The actual label is the correct digit from the dataset.
The predicted label is the digit selected by the model.

For example, actual 9 and predicted 3 means the model
incorrectly classified an image of nine as three.

Dataset indices identify individual images, allowing us
to compare mistakes on the same examples.

### Original Model Errors

The original model made ten validation mistakes.

Five involved images whose actual label was nine:

- Two were predicted as three.
- One was predicted as eight.
- One was predicted as seven.
- One was predicted as one.

The other mistakes involved actual digits two, five,
and eight.

These counts identify patterns but do not establish
why the images were misclassified.

### Wider Model Errors

The wider model made seven validation mistakes:

- Actual 9 → predicted 5: two images.
- Actual 5 → predicted 4: one image.
- Actual 8 → predicted 5: one image.
- Actual 1 → predicted 4: one image.
- Actual 6 → predicted 1: one image.
- Actual 2 → predicted 1: one image.

### Comparing Individual Mistakes

The wider model fixed seven original mistakes.

Fixed image indices:

942, 891, 1646, 275, 719, 1581, and 1274.

Three images remained incorrect in both models:

- Index 1580: actual 9; prediction changed from 8 to 5.
- Index 678: actual 5; both predicted 4.
- Index 77: actual 2; both predicted 1.

The wider model introduced four new mistakes:

- Index 899: actual 8 → predicted 5.
- Index 1747: actual 1 → predicted 4.
- Index 1551: actual 6 → predicted 1.
- Index 1665: actual 9 → predicted 5.

Total wider-model errors:

10 original errors − 7 fixed errors + 4 new errors = 7.

Higher overall accuracy can still include new mistakes.

### Accuracy and Parameter Tradeoff

The wider model used approximately 126% more parameters
to produce three additional correct validation predictions.

More parameters provide greater capacity but require more
learned values to store. This wider architecture also
requires more arithmetic during a forward pass.

The experiment did not measure the wider model's saved
size, runtime memory, or inference latency.

The wider model had higher validation accuracy, while
the original model had fewer parameters.

### Saved Results

Results are stored in results/architecture_comparison.

- validation_comparison.json.
- original_cnn_validation_errors.csv.
- wider_cnn_validation_errors.csv.
- original_cnn_best_validation.pt.
- wider_cnn_best_validation.pt.

The JSON report includes training histories, parameter
counts, validation accuracy, and confusion matrices.

Confusion-matrix rows represent actual labels.
Columns represent predicted labels.

### Findings

Increasing channel counts improved validation accuracy
in this experiment, but more than doubled parameter count.

The wider model fixed some mistakes and introduced others.

Validation accuracy and error analysis describe different
aspects of performance and should be considered together.

### Limitations

The comparison used one training seed and one data split.

Three additional correct predictions do not establish
consistent superiority across other runs or datasets.

Repeated decisions using the same validation set can
also overfit development choices to that set.

No test evaluation occurred during this experiment.

No wider-model inference benchmark was completed.

The results do not establish performance on Qualcomm
hardware.

### What I Can Explain

I can explain how increasing channels adds feature maps,
weights, biases, and model capacity.

I can calculate why the wider model needs 128 inputs
to its first Linear layer.

I can compare validation accuracy and parameter count
and explain the tradeoff.

I can compare actual and predicted labels to analyze
mistakes across individual images.

I can explain why architecture decisions use validation
results rather than test results.